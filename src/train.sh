python train_agent.py  --mode IID   --cogitation_model causal --graph collider  --noise_objects 0    &
python train_agent.py  --mode OOD-S --cogitation_model causal --graph collider  --noise_objects 0    &
python train_agent.py  --mode IID   --cogitation_model causal --graph chain     --noise_objects 0    &
python train_agent.py  --mode OOD-S --cogitation_model causal --graph chain     --noise_objects 0    &
python train_agent.py  --mode IID   --cogitation_model causal --graph full      --noise_objects 0    &
python train_agent.py  --mode OOD-S --cogitation_model causal --graph full      --noise_objects 0    &
python train_agent.py  --mode IID   --cogitation_model causal --graph jungle    --noise_objects 0    &
python train_agent.py  --mode OOD-S --cogitation_model causal --graph jungle    --noise_objects 0    &