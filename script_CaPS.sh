# ER
python3 main_CaPS.py --dataset synthetic --num_nodes 2 --num_samples 1000 --run 10
python3 main_CaPS.py --dataset synthetic --num_nodes 3 --num_samples 1000 --run 10
python3 main_CaPS.py --dataset synthetic --num_nodes 5 --num_samples 1000 --run 10
python3 main_CaPS.py --dataset synthetic --num_nodes 10 --num_samples 1000 --run 10
python3 main_CaPS.py --dataset synthetic --num_nodes 30 --num_samples 1000 --run 10
python3 main_CaPS.py --dataset synthetic --num_nodes 50 --num_samples 1000 --run 10
python3 main_CaPS.py --dataset synthetic --num_nodes 100 --num_samples 1000 --run 10
#physics,sachs
python3 main_CaPS.py --dataset physics --num_samples 5000
python3 main_CaPS.py --dataset sachs_nocycle --num_samples 7466 
#bnlearn nonlinear
python3 main_CaPS.py --dataset ecoli70 --num_samples 10000 --batch_size 1000
python3 main_CaPS.py --dataset magic-niab --num_samples 10000 --batch_size 1000
python3 main_CaPS.py --dataset magic-irri --num_samples 10000 --batch_size 1000
python3 main_CaPS.py --dataset arth150 --num_samples 10000 --batch_size 300
#bnlearn linear
python3 main_CaPS.py --dataset ecoli70_linear --num_samples 10000 --batch 1000
python3 main_CaPS.py --dataset magic-niab_linear --num_samples 10000 --batch_size 1000
python3 main_CaPS.py --dataset magic-irri_linear --num_samples 10000 --batch_size 1000
python3 main_CaPS.py --dataset arth150_linear --num_samples 10000 --batch_size 300