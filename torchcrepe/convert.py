from scipy.stats import triang
import torch

import torchcrepe


###############################################################################
# Pitch unit conversions
###############################################################################


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    cents = torchcrepe.CENTS_PER_BIN * bins + 1997.3794084376191

    # Trade quantization error for noise
    #return dither(cents) # TODO: make dither work in torchscript
    return cents


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))


def cents_to_bins(cents, quantize_fn:torch.func =torch.floor):
    """Converts cents to pitch bins"""
    bins = (cents - 1997.3794084376191) / torchcrepe.CENTS_PER_BIN
    return quantize_fn(bins).int()


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)

def frequency_to_bins_floor(frequency):
    """Convert frequency in Hz to pitch bins"""
    cents = frequency_to_cents(frequency)
    bins = (cents - 1997.3794084376191) / torchcrepe.CENTS_PER_BIN
    return torch.floor(bins).int()

def frequency_to_bins_ceil(frequency):
    """Convert frequency in Hz to pitch bins"""
    cents = frequency_to_cents(frequency)
    bins = (cents - 1997.3794084376191) / torchcrepe.CENTS_PER_BIN
    return torch.ceil(bins).int()


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return 1200 * torch.log2(frequency / 10.)


###############################################################################
# Utilities
###############################################################################


def dither(cents):
    """Dither the predicted pitch in cents to remove quantization error"""
    noise = triang.rvs(c=0.5,
                                   loc=-torchcrepe.CENTS_PER_BIN,
                                   scale=2 * torchcrepe.CENTS_PER_BIN,
                                   size=cents.size())
    return cents + cents.new_tensor(noise)
