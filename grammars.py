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
    p_orc_local: np.float64,
    p_one_arg: np.float64,
    p_adj_interveners: np.float64,
    p_one_adj: np.float64
) -> str:
    return f"""
    RC -> SRC [{p_src}] | ORC [{1-p_src}]
    SRC -> 'RPNom' 'V' ArgSRC [{p_src_local*(1-p_adj_interveners)}] | 'RPNom' ArgSRC 'V'  [{(1-p_src_local)*(1-p_adj_interveners)}] | 'RPNom' AdjIntv 'V' ArgSRC [{p_adj_interveners*p_src_local}] | 'RPNom' AdjIntv ArgSRC 'V' [{p_adj_interveners*(1-p_src_local)}]
    ArgSRC -> 'DO' [{p_one_arg}] | 'DO' 'IO' [{1-p_one_arg}]
    ORC -> 'RPAcc' 'V' ArgORC [{p_orc_local*(1-p_adj_interveners)}] | 'RPAcc' ArgORC 'V'  [{(1-p_orc_local)*(1-p_adj_interveners)}] | 'RPAcc' AdjIntv 'V' ArgORC [{p_adj_interveners*p_orc_local}] | 'RPAcc' AdjIntv ArgORC 'V' [{p_adj_interveners*(1-p_orc_local)}]
    ArgORC -> 'Subj' [{p_one_arg}] | 'Subj' 'IO' [{1-p_one_arg}]
    AdjIntv -> 'Adj' [{p_one_adj}] | 'Adj' 'Adj' [{(1-p_one_adj)}]
    """