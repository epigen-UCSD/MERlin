import numpy as np

def test_codebook_get_bit_count(simple_merfish_data):
    assert simple_merfish_data.get_codebook().get_bit_count() == 16

def test_codebook_get_bit_names(simple_merfish_data):
    for i,n in enumerate(simple_merfish_data.get_codebook().get_bit_names()):
        assert n == 'bit' + str(i+1)

def test_codebook_get_barcodes(simple_merfish_data):
    bcSetWithBlanks = simple_merfish_data.get_codebook().get_barcodes()
    assert len(bcSetWithBlanks) == 140
    assert all([len(x)==16 for x in bcSetWithBlanks])
    assert all([np.sum(x)==4 for x in bcSetWithBlanks])
    bcSetNoBlanks = simple_merfish_data.get_codebook().get_barcodes(
            ignoreBlanks=True)
    assert len(bcSetNoBlanks) == 70
    assert all([len(x)==16 for x in bcSetNoBlanks])
    assert all([np.sum(x)==4 for x in bcSetNoBlanks])