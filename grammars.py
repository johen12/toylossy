import numpy as np

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
    SRC -> 'RPNom' 'V' ArgSRC [{p_src_local*(1-p_adj_interveners)}] | 'RPNom' ArgSRC 'V'  [{(1-p_src_local)*(1-p_adj_interveners)}] | 'RPNom' AdjIntv 'V' ArgSRC [{p_adj_interveners*p_src_local}] | 'RPNom' ArgORC AdjIntv 'V' [{p_adj_interveners*(1-p_src_local)}]
    ArgSRC -> 'DO' [{p_one_arg}] | 'DO' 'IO' [{1-p_one_arg}]
    ORC -> 'RPAcc' 'V' ArgORC [{p_orc_local*(1-p_adj_interveners)}] | 'RPAcc' ArgORC 'V'  [{(1-p_orc_local)*(1-p_adj_interveners)}] | 'RPAcc' AdjIntv 'V' ArgORC [{p_adj_interveners*p_orc_local}] | 'RPAcc' ArgORC AdjIntv 'V' [{p_adj_interveners*(1-p_orc_local)}]
    ArgORC -> 'Subj' [{p_one_arg}] | 'Subj' 'IO' [{1-p_one_arg}]
    AdjIntv -> 'Adj' [{p_one_adj}] | 'Adj' 'Adj' [{(1-p_one_adj)}]
    """

def gen_hindi_grammar_exp1(
    p_src: np.float64, 
    p_src_local: np.float64,
    p_obj_elision: np.float64,
    p_orc_local: np.float64,
    p_subj_elision: np.float64,
) -> str:
    return f"""
    RC -> SRC [{p_src}] | ORC [{1-p_src}]
    SRC -> 'RPErg' InnerSRC [{1-p_obj_elision}] | 'RPErg' 'V' [{p_obj_elision}]
    InnerSRC -> 'DO' 'V' [{1-p_src_local}] | 'V' 'DO' [{p_src_local}]
    ORC -> 'RPAcc' InnerORC [{1-p_subj_elision}] | 'RPAcc' 'V' [{p_subj_elision}]
    InnerORC -> 'Subj' 'V' [{1-p_src_local}] | 'V' 'Subj' [{p_src_local}]
    """

def gen_hindi_grammar_exp2(
    p_cp: np.float64,
    p_cp_short: np.float64,
    p_cp_lightverb: np.float64,
    p_sp_short: np.float64,
    p_sp_lightverb: np.float64,
) -> str:
    return f"""
    S -> CPP [{p_cp}] | SPP [{1-p_cp}]
    CPP -> 'CPNoun' 'Adj' CPVerb [{p_cp_short}] | 'CPNoun' 'Adj' 'Adj' CPVerb [{1-p_cp_short}]
    CPVerb -> 'LightVerb' [{p_cp_lightverb}] | 'OtherVerb' [{1-p_cp_lightverb}]
    SPP -> 'SPNoun' 'Adj' SPVerb [{p_sp_short}] | 'SPNoun' 'Adj' 'Adj' SPVerb [{1-p_sp_short}]
    SPVerb -> 'LightVerb' [{p_sp_lightverb}] | 'OtherVerb' [{1-p_sp_lightverb}]
    """