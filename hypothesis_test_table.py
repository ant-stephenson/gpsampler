#%%
import numpy as np
import pandas as pd
import re

from results_utils import style_to_latex

filepath = "experiments/sampling/ciq_hypothesis_tests.csv"
# filepath = "dataset_test_calibration.csv"

res = pd.read_csv(filepath)
res.replace({"ciq": "CIQ", "rbf": "RBF", "exp": "Exp"}, inplace=True)
indices = ["method","kernel","d","ls"]
if "nu" in res.columns:
    indices += ["nu"]
res = res.sort_values(["method","kernel","d","ls","m"]).set_index(indices)

res.drop(["nreps","id"],axis=1,inplace=True)
if "calibration" not in filepath:
    res.drop(["m"],axis=1,inplace=True)
else:
    res.set_index(["m"], append=True, inplace=True)

res = res.loc[~res.index.duplicated(keep='first')]

#%%
def boldify_sig(col, thresh=0.1, comparison_op = '<='):
    if comparison_op == '<=':
        return (col <= thresh).map({True: 'font-weight: bold', False: ''})
    elif comparison_op == '>=':
        return (col >= thresh).map({True: 'font-weight: bold', False: ''})
    elif comparison_op == '>':
        return (col > thresh).map({True: 'font-weight: bold', False: ''})
    elif comparison_op == '<':
        return (col < thresh).map({True: 'font-weight: bold', False: ''})
    elif comparison_op == '==':
        return (col == thresh).map({True: 'font-weight: bold', False: ''})
    else:
        raise ValueError(f"Unsupprted comparison operator {comparison_op}.")


    # print(res.drop("RBF",axis=0,level="kernel").sort_values(["pval","reject"],ascending=[False,True]))
# res.sort_values(["pval","reject"],ascending=[False,True], inplace=True)
style = res.style.format(precision=3).format_index(precision=1).format(subset=["reject"],precision=1)
style.apply(lambda x: boldify_sig(x, thresh=0.1, comparison_op='<='), axis=0,subset=["reject"])
style.apply(lambda x: boldify_sig(x, thresh=0.1, comparison_op='>'), axis=0,subset=["pval", "kspval"])
style

# with pd.option_context('display.float_format', '{:0.1f}'.format):
#     print(style)

#%%

latex = style_to_latex(style)

latex = re.sub(r"\\textbf{pval}",r"$\\langle p \\rangle$", latex)
latex = re.sub(r"\\textbf{kspval}",r"$p_{KS}$", latex)
latex = re.sub(r"\\textbf{stat}",r"$\\langle T \\rangle$", latex)
latex = re.sub(r"\\textbf{ksstat}",r"$T_{KS}$", latex)
latex = re.sub(r"\\textbf{reject}",r"$r$", latex)

print(latex)
# %%
