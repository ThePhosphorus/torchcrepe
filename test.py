import torchcrepe
import torch

from typing import List, Optional

class TorchCrepeTest(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sr = 16000
        self.window = 160
        self.device = "cpu"
        self.crepe = torchcrepe.TorchCrepe('full', self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * torch.log(1 + f0_min / 700)
        f0_mel_max = 1127 * torch.log(1 + f0_max / 700)
        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        # Compute pitch using first gpu
        audio = torch.tensor(x)[None].float()
        f0, pd =  self.predict(
            audio=audio, 
            fmin=float(f0_min), 
            fmax=float(f0_max),  
            batch_size=batch_size)
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0

        return f0

    def predict(self, audio: torch.Tensor,
            fmin: float=50.,
            fmax: float=2006.0,
            batch_size: Optional[int]=None,
            pad: bool=True) :
        pitch_result: List[torch.Tensor] = []
        periodicity_result: List[torch.Tensor] = []
        PITCH_BINS = 360

        # Postprocessing breaks gradients, so just don't compute them
        with torch.no_grad():

            # Preprocess audio
            generator = torchcrepe.preprocess(audio,
                                self.sr,
                                self.window,
                                batch_size,
                                self.device,
                                pad)
            for frames in generator:

                # Infer independent probabilities for each pitch bin
                probabilities = self.crepe(frames, embed=False)

                # shape=(batch, 360, time / hop_length)
                probabilities = probabilities.reshape(
                    audio.size(0), -1, PITCH_BINS).transpose(1, 2)

                # Convert probabilities to F0 and periodicity
                pitch, periodicity = torchcrepe.postprocess(probabilities,
                                    fmin,
                                    fmax)

                # Place on same device as audio to allow very long inputs
                pitch_result.append(pitch.to(audio.device))
                periodicity_result.append(periodicity.to(audio.device))
        return torch.cat(pitch_result, 1), torch.cat(periodicity_result, 1)
        


model = TorchCrepeTest()

script = torch.jit.script(model)

print("test")
