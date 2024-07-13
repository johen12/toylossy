import numpy as np

def gen_russian_grammar_exp1(
        p_src: np.float64, 
        p_src_local: np.float64,
        p_orc_canonical: np.float64
) -> str:
    return f"""
    RC -> SRC [{p_src}] | ORC [{1-p_src}]
    SRC -> 'RPNom' 'V'  [{p_src_local}] | 'RPNom' 'NP' 'V'  [{1-p_src_local}]
    ORC -> 'RPAcc' 'NP' 'V' [{p_orc_canonical}] | 'RPAcc' 'V' [{1-p_orc_canonical}]
    """

def gen_russian_grammar_exp2(
    p_src: np.float64, 
    p_src_local: np.float64,
    p_orc_non_local: np.float64,
    p_src_one_arg: np.float64,
    p_orc_one_arg: np.float64
) -> str:
    return f"""
    RC -> SRC [{p_src}] | ORC [{1-p_src}]
    SRC -> 'RPNom' 'V' ArgSRC [{p_src_local}] | 'RPNom' ArgSRC 'V'  [{1-p_src_local}]
    ArgSRC -> 'DO' [{p_src_one_arg}] | 'DO' 'IO' [{1-p_src_one_arg}]
    ORC -> 'RPAcc' ArgORC 'V' [{p_orc_non_local}] | 'RPAcc' 'V' ArgORC [{1-p_orc_non_local}]
    ArgORC -> 'Subj' [{p_orc_one_arg}] | 'Subj' 'IO' [{1-p_orc_one_arg}]
    """