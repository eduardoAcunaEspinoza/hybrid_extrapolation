from typing import Dict
import torch
import torch.nn as nn
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.shm import SHM
from neuralhydrology.modelzoo.hbv import HBV
from neuralhydrology.modelzoo.uhrouting import UHRouting


class HybridModel(BaseModel):
    """Wrapper to combine a deep learning model with a conceptual hydrological models.

    In the current implementation, the deep learning model is always an LSTM. The conceptual model is configurable using
    the config argument `conceptual_model`. Currently supported are `['SHM' and 'HBV']`.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(HybridModel, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)

        self.conceptual_model = HybridModel._get_conceptual_model(cfg=cfg)
        self.n_conceptual_model_params = len(self.conceptual_model.parameter_ranges) * cfg.n_conceptual_models

        self.conceptual_routing = cfg.conceptual_routing
        if self.conceptual_routing:
            self.routing_model = UHRouting(cfg=cfg)
            self.n_routing_params = len(self.routing_model.parameter_ranges)
        else:
            self.n_routing_params = 0

        self.linear = nn.Linear(
            in_features=cfg.hidden_size, out_features=self.n_conceptual_model_params + self.n_routing_params
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size : 2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs, dynamic parameters and intermediate states coming from the conceptual model.
        """
        # run lstm
        x_d = self.embedding_net(data)
        lstm_output, _ = self.lstm(input=x_d)
        lstm_output = lstm_output.transpose(0, 1)  # reshape to [batch_size, seq, n_hiddens]

        # map lstm outputs to the dimension of the conceptual model´s parameters
        lstm_output = self.linear(lstm_output)

        # map lstm output to parameters of conceptual model
        warmup_period = self.cfg.seq_length - self.cfg.predict_last_n
        parameters_warmup, parameters_simulation = self.conceptual_model.map_parameters_conceptual(
            lstm_out=lstm_output[:, :, : self.n_conceptual_model_params], warmup_period=warmup_period
        )
        # run conceptual model: warmup
        with torch.no_grad():
            pred = self.conceptual_model(x_conceptual=data["x_d_c"][:, :warmup_period, :], parameters=parameters_warmup)

        # run conceptual model: simulation
        pred = self.conceptual_model(
            x_conceptual=data["x_d_c"][:, warmup_period:, :],
            parameters=parameters_simulation,
            initial_states=pred["final_states"],
        )

        if self.conceptual_routing:
            # map lstm output to parameters of routing model
            _, parameters_simulation = self.routing_model.map_parameters_conceptual(
                lstm_out=lstm_output[:, :, self.n_conceptual_model_params :], warmup_period=warmup_period
            )

            # apply routing routine
            pred["y_hat"] = self.routing_model(discharge=pred["y_hat"], parameters=parameters_simulation)

        return pred

    @staticmethod
    def _get_conceptual_model(cfg: Config) -> BaseConceptualModel:
        """Get conceptual model, depending on the run configuration.

        Parameters
        ----------
        cfg : Config
            The run configuration.

        Returns
        -------
        BaseConceptualModel
            A new conceptual model instance of the type specified in the config.
        """
        if cfg.conceptual_model.lower() == "shm":
            conceptual_model = SHM(cfg=cfg)
        elif cfg.conceptual_model.lower() == "hbv":
            conceptual_model = HBV(cfg=cfg)
        else:
            raise NotImplementedError(
                f"{cfg.conceptual_model} not implemented or not linked in `_get_conceptual_model()`"
            )

        return conceptual_model
