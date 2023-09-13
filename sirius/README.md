Real-robot datasets accompanying the Sirius dataset. The dataset comprises two tasks,
kcup and gear. The kcup task requires opening the kcup holder, inserting the kcup into the holder,
and closing the holder. The gear task requires inserting the blue gear onto the right peg,
followed by inserting the smaller red gear.

We use a 7-DoF Franka Emika Panda robot which is operated via Operational Space Control
(OSC). The robot is either controlled either by a learned policy or a human operator who is involved
during initial demonstration collection or to intervene when the policy exhibits undesirable behaviors.
We label each timestep of data according to the action mode (initial demo, robot, human intervention).
Additionally, the `intv_label` key labels the 15 timesteps preceding human intervention as 
"pre-interventions" which represent abnormal states that should be avoided. In Sirius, we utilize an intervention-guided imitation learning algorithm, where we assign higher weights to intervention
and lower weights to pre-intervention states.

Link to dataset: https://utexas.app.box.com/s/htjpowji1ynukkzge0wz50p1lmtbo0m4
