# Data

This folder contains the methods to load the dataset and the tokenizers.

## Files

### Code

* `__init__.py`: python module
* `augmentation_base.py`: base functions for data augmentation
* `augmentation_bi.py`: data augmentation for bilingual data
* `augmentation_mono.py`: data augmentation for monolingual data
* `data.py`: functions for loading the datasets
* `pos.py`: functions for the POS tags and grammatical features
* `tokenizers.py`: tokenizer classes
* `utils.py`: regex used to process the data, dictionaries for transliteration

### Data

* `CIEP_pymupdf.csv`: (intermediate file) CIEP data extracted by PyMuPDF
* `ETP_fix.csv`: (intermediate file) ETP data
* `Etruscan.csv`: Etruscan data with all the inscriptions (including those without translation)
* `ETP_POS.csv`: ETP vocabulary with grammatical categories, POS tags and translations
* `ETPNames.txt`: (raw data) list of Etruscan proper names, their translation and grammatical features
* `ETPSuff.txt`: (raw data) list of Etruscan suffixes
* `ETPWords.txt`: (raw data) list of Etruscan words, their translation and grammatical features

### Others

* `all_small_char.model`: character tokenizer (SentencePiece)
* `all_small_char.vocab`: vocaulary of the character tokenizer (SentencePiece)
* `all_small_word.model`: word tokenizer (SentencePiece)
* `all_small_word.vocab`: vocaulary of the word tokenizer (SentencePiece)
* `Notebooks`: some notebooks used for the early work
