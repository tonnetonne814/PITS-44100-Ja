""" from https://github.com/keithito/tacotron """
import re
from unicodedata import normalize

from text.cleaners import collapse_whitespace
from text.symbols import lang_to_dict, lang_to_dict_inverse


def text_to_sequence(raw_text, lang):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        text: string to convert to a sequence
        lang: language of the input text
    Returns:
        List of integers corresponding to the symbols in the text
    '''

    _symbol_to_id = lang_to_dict(lang)
    text = collapse_whitespace(raw_text)

    if lang == 'ko_KR':    
        text = normalize('NFKD', text)
        sequence = [_symbol_to_id[symbol] for symbol in text]
        tone = [0 for i in sequence]

    elif lang == 'en_US':
        _curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')
        sequence = []

        while len(text):
            m = _curly_re.match(text)

            if m is not None:
                ar = m.group(1)
                sequence += [_symbol_to_id[symbol] for symbol in ar]
                ar = m.group(2)
                sequence += [_symbol_to_id[symbol] for symbol in ar.split()]
                text = m.group(3)
            else:
                sequence += [_symbol_to_id[symbol] for symbol in text]
                break

        tone = [0 for i in sequence]

    ### Add ###
    elif lang == 'pyopenjtalk_prosody':
        sequence = []
        phonomes = pyopenjtalk_g2p_prosody(text)                    # 文章⇒音素
        sequence += [_symbol_to_id[symbol] for symbol in phonomes]  # 音素⇒index番号
        tone = [0 for i in sequence]                                # 音素分だけtoneも追加(なにこれ?)
    ###########

    else:
        raise RuntimeError('Wrong type of lang')

    assert len(sequence) == len(tone)
    return sequence, tone


def sequence_to_text(sequence, lang):
    '''Converts a sequence of IDs back to a string'''
    _id_to_symbol = lang_to_dict_inverse(lang)
    result = ''
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text

### Add from espnet ### 
# ESPNet:https://github.com/espnet/espnet
#######################
def pyopenjtalk_g2p_prosody(text: str, drop_unvoiced_vowels: bool = True) :
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    labels = _extract_fullcontext_label(text)
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones

from packaging.version import parse as V
def _extract_fullcontext_label(text):
    import pyopenjtalk

    if V(pyopenjtalk.__version__) >= V("0.3.0"):
        return pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    else:
        return pyopenjtalk.run_frontend(text)[1]


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))

#######################