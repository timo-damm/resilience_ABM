A multi-level Agent Based Model (ABM) of resilience at the individual and group level. Tailored to social movement organisations (SMOs), and based on extended fieldwork with an SMO. Find more details about the fieldwork [here](https://osf.io/9jbyr/overview?view_only=66495dfcfc2d4f00a2c9c20b89fe1fb6)

more descriptions to follow

## model overview:
_environment_:
- stressors (macro level) (level and changes vary by scenario)
- external social support (micro level) (distribution varies by scenario)

_variables_:
- internal social support (meso level) 
- SMO resilience (meso level) 
- causes of burnout (all levels) 
- individual resilience (symptoms of burnout) (micro level) 

_mechanisms_:
- agents dropping out: threshold 
- agents joining (not operational yet): rate proportional to repression, based on parameterisation, social support drawn from current distribution of the group
- edge updating: newcomers joining will make 3 connections, for everyone else, probabilistic edge updating with anyone in group, probability influenced by individual level of burnout, also include edge deletion with low probability

- environmental stress negatively influences resilience at micro and meso level
- outside of social support positively influences resilience at micro level
- causes of burnout have negative feedback loop with individual resilience 
- internal social support has positive feedback with resilience at micro and meso level
- resilience at micro and meso level has positive feedback loop with each other
