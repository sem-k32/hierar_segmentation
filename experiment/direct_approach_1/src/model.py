import torch
import torch.nn as nn


class encoderBlock(nn.Module):
    def __init__(
            self,
            chanels_in: int, 
            chanels_out: int
    ) -> None:
        """increase number of channels,apply several conv+relu transforms, max pool

        Args:
            chanels_in (int): _description_
            chanels_out (int): _description_
        """
        super().__init__()

        self._chanels_in = chanels_in
        self._chanels_out = chanels_out

        NUM_CONV_LAYS = 3
        CONV_SIZE = (3, 3)
        pad_size = [(conv_size_el - 1) // 2 for conv_size_el in CONV_SIZE]
        conv_seq = []
        # first conv layer
        conv_seq.append(nn.Conv2d(
                        chanels_in,
                        chanels_out,
                        CONV_SIZE,
                        padding=pad_size,
                        padding_mode="reflect"
        ))
        conv_seq.append(
            nn.ReLU()
        )
        # rest conv layers
        for _ in range(NUM_CONV_LAYS - 1):
            conv_seq.append(nn.Conv2d(
                        chanels_out,
                        chanels_out,
                        CONV_SIZE,
                        padding=pad_size,
                        padding_mode="reflect"
            ))
            conv_seq.append(
                nn.ReLU()
            )
        
        POOL_SIZE = (2, 2)
        self._pooling = nn.MaxPool2d(POOL_SIZE)

        # final transform sequence
        self._conv_seq = nn.Sequential(*conv_seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # non-pooled feature map is used by decoder
        output = self._conv_seq(x)
        x = self._pooling(torch.clone(output))

        return x, output


class decoderBlock(nn.Module):
    def __init__(
            self,
            chanels_in: int, 
            chanels_out: int
    ) -> None:
        """upsample feature map, apply several conv+relu transforms, decrease num of channels

        Args:
            chanels_in (int): _description_
            chanels_out (int): _description_
        """
        super().__init__()

        self._chanels_in = chanels_in
        self._chanels_out = chanels_out

        UPSAMPLE_SCALE = (2, 2)
        self._upsampling = nn.Upsample(
            scale_factor=UPSAMPLE_SCALE,
            mode="bilinear"
        )

        NUM_CONV_LAYS = 3
        CONV_SIZE = (3, 3)
        pad_size = [(conv_size_el - 1) // 2  for conv_size_el in CONV_SIZE]
        conv_seq = []
        # first conv layer counts encoder output
        conv_seq.append(nn.Conv2d(
                        2 * chanels_in,
                        chanels_in,
                        CONV_SIZE,
                        padding=pad_size,
                        padding_mode="reflect"
        ))
        conv_seq.append(
            nn.ReLU()
        )
        # middle conv layers
        for _ in range(NUM_CONV_LAYS - 2):
            conv_seq.append(nn.Conv2d(
                        chanels_in,
                        chanels_in,
                        CONV_SIZE,
                        padding=pad_size,
                        padding_mode="reflect"
            ))
            conv_seq.append(
                nn.ReLU()
            )
        # last conv layer with channels reduction
        conv_seq.append(nn.Conv2d(
                        chanels_in,
                        chanels_out,
                        CONV_SIZE,
                        padding=pad_size,
                        padding_mode="reflect"
        ))
        conv_seq.append(
            nn.ReLU()
        )
        self._conv_seq = nn.Sequential(*(conv_seq))

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self._upsampling(x)
        return self._conv_seq(torch.concat([encoder_output, x], dim=1))


class directSegmentator(nn.Module):
    def __init__(
            self,
            num_levels: int,
            num_classes: int
    ) -> None:
        """ implementing UNet-like architecture for direct image segmentation, without hiearcal context.
             Resolution decreases at 2 each level, num of channels is doubling with each level.
             logits are output of the model

        Args:
            num_levels (int): depth of the encoder-decoder
            num_classes (int): num of final classes
        """
        super().__init__()

        self._encoder_list = nn.ModuleList()
        self._decoder_list = nn.ModuleList()

        # number of channels is doubling with each level
        CHANNELS_START = 16
        # first encoder-decoder layer
        self._encoder_list.append(encoderBlock(chanels_in=3, chanels_out=CHANNELS_START))
        self._decoder_list.append(decoderBlock(chanels_in=CHANNELS_START, chanels_out=num_classes))
        for i in range(num_levels - 1):
            self._encoder_list.append(
                encoderBlock(CHANNELS_START * (2 ** i), CHANNELS_START * (2 ** (i + 1)))
            )
            self._decoder_list.append(
                decoderBlock(CHANNELS_START * (2 ** (i + 1)), CHANNELS_START * (2 ** i))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = []
        # encoder stack
        for i in range(len(self._encoder_list)):
            x, encoder_out = self._encoder_list[i](x)
            encoder_outputs.append(encoder_out)
        # decoder stack
        for i in range(len(self._decoder_list) - 1, 0 - 1, -1):
            x = self._decoder_list[i](x, encoder_outputs[i])
            
        return x
        


