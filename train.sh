parallel -j 3 python train.py --swanlab --decouple --train_step 20 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.05 --action_diff --use_expert --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 40 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.05 --action_diff --use_expert --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 80 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.05 --action_diff --use_expert --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 20 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.1 --action_diff --use_expert --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 40 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.1 --action_diff --use_expert --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 80 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.1 --action_diff --use_expert --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 20 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.05 --action_diff --use_expert --use_kl --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 40 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.05 --action_diff --use_expert --use_kl --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 80 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.05 --action_diff --use_expert --use_kl --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 20 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.1 --action_diff --use_expert --use_kl --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 40 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.1 --action_diff --use_expert --use_kl --seed {1} ::: 1 2 3
parallel -j 3 python train.py --swanlab --decouple --train_step 80 --loop_nums 5 --adv_steps 6 --cuda_number 1 --attack_eps 0.1 --action_diff --use_expert --use_kl --seed {1} ::: 1 2 3