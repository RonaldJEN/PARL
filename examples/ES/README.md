## How to use
### Dependencies
+ python2.7 or python3.5+
+ [paddlepaddle==1.3.0](https://github.com/PaddlePaddle/Paddle)
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym==0.9.4
+ mujoco-py==0.5.1


### Distributed Training

#### Learner
```sh
python train.py 
```

#### Actors
```sh
sh run_actors.sh
```

If you want to use different number of actors, please modify `actor_num` in es_config.py and run_actors.sh.
