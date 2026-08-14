# Reflective Essay

The results of this project are mostly negative, and to be honest that disappointed me. I can reflect on it because the negative results trace back to decisions I made long before any experiment was run.

The project began in a single direction, built around generators, consumers, and storage, with the battery acting as the smart device that balances profit against its own health. The motivation then became adapting to scenarios that are not known at training time.

## Strengths

What I do still stand behind is the motivation. It addresses a real problem in today's energy storage industry, where assets are exposed to shocks and changing conditions.

The results are also not completely negative. The dyna results (Sutton, 1991) are positive, though under one condition, which is that the world model has to pass the calibration gate first. When it does, an agent trained on a single year of observed prices comes close to what a model-free agent achieves with an unlimited supply of real ones, because the model turns that year into far more training days than the year itself contains. The operator needs more history, but quality plays a crucial role.

The part of the work I am most satisfied with is the effort that went into establishing why each result came out the way it did. The obvious explanation for dyna's margin is that it simply takes more gradient steps, so the comparison was built to rule that out. Replay trains on the identical year for as many epochs as it wants and still finishes far behind, which leaves the number of distinct training days as the thing doing the work. The negative results were handled the same way. The nightly refit loop fails because incremental refitting drives the market model out of calibration within three nights, weeks before the shift it was meant to catch, and the physics-informed surrogate loses to a plain network because an ablation against deliberately wrong physics shows its residual carries no information at small data budgets. A negative result with a named mechanism is a positive find rather than just a run that didn't work.

Though I am careful about how far that goes. The gate's contribution is based on the intuition that the training data generated must be similar to the real data, but because no policy was ever trained inside a model the gate rejected, that claim is not proven. So every dyna number I quote comes from the one draw that passed.

## Weaknesses

At the start I spent too much time trying to devise something interesting and complicated out of the things I wanted to work with. I was too ambitious about what the project should be, and the direction I settled on was never concrete enough, so it went through many refinements. The project became an amalgamation of different techniques and approaches, each of which introduced its own problems and challenges, and together they led to poor results. I should have been more decisive about which project to work on and kept it simple and focused on the things I wanted to work with. The project would then have shaped itself once the foundation was right.

The clearest weakness in how I worked was that I did not ask why something worked when it did. The market model has to pass a calibration gate before it is allowed to train a policy, and at a budget of 365 observed days it passed, so I treated that as settled and built the rest of the project on that configuration. Only much later, when I repeated the same fit over five independent draws of the year, did I find that it passes once in five. The setting I had been treating as the working case was a favourable draw, and everything downstream had inherited that luck. The direction survived the check, because the dyna vs replay ordering still held at every training seed once the history was pinned, but the headline margin fell by about half. A single passing run tells you that a method can work, not that it is flawless, and the sweep that separates those two costs a few hours of compute. I would rather have paid that in the second month than in the final weeks.

## Theory and Practical Work

Building the simulator taught me a great deal about the energy market and about battery dynamics, and that knowledge went directly into the environment and tools the models train in. Because I have a lot of programming experience, the simulation itself was not the hard part, it was designing the experiments and implementing the training techniques that took the most time.

The stage is a battery that has to balance profit against its own health, so something has to model both the market and the battery physics. A physics-informed network (Raissi et al., 2019) is the intuitive choice for the battery half, because charging and discharging obey laws I already know, and writing those laws into the model should pay off exactly where samples are scarce, which is the sample efficiency problem I set out to tackle in the first place. It did not work out that way. A plain MLP won at every budget below 500 samples, and by widening margins as the data thinned. What decides whether a physics prior helps is not whether the system is physical, it is whether the function is hard to learn from the data available. The battery map is three inputs to three outputs and smooth, so an unconstrained network interpolates it from a handful of points and there is nothing left for the prior to contribute.

## Further Work

## What I Would Have Done With More Time

## Legal, Social, Ethical and Sustainability Issues

## References

Sutton, R. S. (1991). Dyna, an integrated architecture for learning, planning, and reacting. *ACM SIGART Bulletin*, 2(4), 160-163. doi:10.1145/122344.122377

Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707. doi:10.1016/j.jcp.2018.10.045
