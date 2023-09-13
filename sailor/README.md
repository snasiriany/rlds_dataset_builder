We collect a large prior dataset of task-agnostic play behaviors involving the food items
and the pot and pan. Overall our prior dataset involves 150 trajectories each with approximately
2,000 timesteps, resulting in approximately 300,000 total timesteps. For each trajectory we first
initialize the scene by randomly sampling four out of eight food items (milk, bread, butter, sausage,
fish, tomato, banana, cheese) and randomly placing these four items around the serving area. We
also randomly initialize the pot and pan on the two front stove burners or occasionally place one on
the table next to the serving area. We then randomly pick and place food items either on the table,
the serving area, or the pot and pan. We also occasionally pick and place the pot or pan to the table
or stove burners.

We also include data for three target tasks: (1) Real-Breakfast: setting up a breakfast table by
placing the bread, butter, and milk in the serving area; (2) Real-Cook: cooking a meal by placing
the fish, sausage, and tomato into the frying pan; (3) Real-Cook-Pan: a variant of the Real-Cook
task involving placing the pan onto the stove. We collect 30 demonstrations for each of these tasks.

We use a 7-DoF Franka Emika Panda robot which is operated via Operational Space Control
(OSC). We found OSC to be a fitting choice, as it offers task-space compliant behavior that
makes for a more intuitive data collection experience. We restrict the OSC controller to the position
and yaw of the end effector, which combined with the gripper controller results in a 5-dimensional
action space. For observations, the agent has access to proprioceptive information consisting of
the robot end effector pose and gripper state, in addition to RGB images from a third-person view
camera and an eye-in-hand camera.

Link to dataset: https://utexas.box.com/s/n8wdjut9tluwq3kqq2yu515v8r45ivfc
