python3 create_figure_2_subplot.py \
--evals_bfw /Users/sahanpaliskara/Documents/cos534/results/bfw/fairface_on_bfw_eval.json \
--evals_laofiw /Users/sahanpaliskara/Documents/cos534/results/LAOFIW/val/fairface_on_laofiw_eval.json /Users/sahanpaliskara/Documents/cos534/results/LAOFIW/val/bfw_on_laofiw_eval.json \
--evals_cc /Users/sahanpaliskara/Documents/cos534/results/cc/gen/fairface_on_cc_eval.json /Users/sahanpaliskara/Documents/cos534/results/cc/gen/bfw_on_cc_eval.json \
--evals_fairface /Users/sahanpaliskara/Documents/cos534/results/fairface/bfw_on_fairface_1_eval.json /Users/sahanpaliskara/Documents/cos534/results/fairface/bfw_on_fairface_2_eval.json /Users/sahanpaliskara/Documents/cos534/results/fairface/bfw_on_fairface_3_eval.json \
--gender_to_idx /Users/sahanpaliskara/Documents/cos534/data/gender.json \
--race_to_idx /Users/sahanpaliskara/Documents/cos534/data/race.json \
--outdir /Users/sahanpaliskara/Documents/cos534/results/fig2 \
--bfw_ann_path /Users/sahanpaliskara/Documents/bfw-v0.1.5-datatable.csv