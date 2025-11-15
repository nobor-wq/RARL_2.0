from gymnasium.envs.registration import register

register(id='TrafficEnv1-v0', entry_point='Environment.environment.env1.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv2-v0', entry_point='Environment.environment.env2.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv3-v0', entry_point='Environment.environment.env3.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False, 'use_gui': False, 'render_mode': None})

register(id='TrafficEnv4-v0', entry_point='Environment.environment.env4.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv1-v1', entry_point='Environment.environment.env1_1.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv2-v1', entry_point='Environment.environment.env2_1.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv3-v1', entry_point='Environment.environment.env3_1.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv4-v1', entry_point='Environment.environment.env4_1.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv1-v2', entry_point='Environment.environment.env1_2.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv2-v2', entry_point='Environment.environment.env2_2.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv3-v2', entry_point='Environment.environment.env3_2.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv4-v2', entry_point='Environment.environment.env4_2.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv5-v0', entry_point='Environment.environment.env5.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False})

register(id='TrafficEnv6-v0', entry_point='Environment.environment.env6.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False, 'clipping':False, 'v_action_obs_flag':False, 'remain_attack_time_flag':False})

register(id='TrafficEnv7-v0', entry_point='Environment.environment.env7.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False, 'use_gui': False, 'render_mode': None})

register(id='TrafficEnv8-v0', entry_point='Environment.environment.env8.traffic_env:Traffic_Env',
         kwargs={'attack': False, 'adv_steps': 2, 'eval': False, 'defense': False, 'use_gui': False, 'render_mode': None})
