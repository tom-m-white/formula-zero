
```text
  __  __           _      _     
 |  \/  | ___   __| | ___| |___ 
 | |\/| |/ _ \ / _` |/ _ \ / __|
 | |  | | (_) | (_) |  __/ \__ \
 |_|  |_|\___/ \__,_|\___|_|___/
```

*This contains information on the models created from training in Formula Zero.*  
*(updated: 3/5/2026 - Tom White)*

## How to install?
1) pip install matplotlib streamlit pandas numba numpy pygame
2) Run Main.py

## Table of Contents
- [Section 1: Models](#section-1-models)
- [Section 2: How each model differs](#section-2-how-each-model-differs)
- [Section 3: Model Stats](#section-3-model-stats)
  - [Easy Difficulty](#easy-difficulty)
  - [Standard Difficulty](#standard-difficulty)
  - [Hard Difficulty](#hard-difficulty)
  - [Insane Difficulty](#insane-difficulty)
  - [Extreme Difficulty](#extreme-difficulty)

---

## Section 1: Models

| # | Filename | Name | Generations |
|---|---|---|---|
| 1 | `alpha_720gen_v2.npz` | alpha | 720 |
| 2 | `alpha_1440gen_v2.npz` | alpha | 1440 |
| 3 | `beta_1280gen_v2.npz` | beta | 1280 |
| 4 | `gamma_200gen_v2.npz` | gamma | 200 |
| 5 | `delta_1050gen_v2.npz` | delta | 1050 |
| 6 | `epsilon_555gen_v2.npz` | epsilon | 555 |
| 7 | `shadow_fury_5252gen.npz` | shadow_fury | 5252 |
| 8 | `shadow_fury_8452gen.npz` | shadow_fury | 8452 |
| 9 | `valkyrie_1267gen.npz` | valkyrie | 1267 |
| 10 | `valkyrie_19767gen.npz` | valkyrie | 19767 |
| 11 | `astraea_400gen.npz` | astraea | 400 |
| 12 | `astraea_1040gen.npz` | astraea | 1040 |
| 13 | `livid_dagger_500gen.npz` | livid_dagger | 500 |
| 14 | `livid_dagger_3500gen.npz` | livid_dagger | 3500 |
| 15 | `the_human_radio_62gen.npz` | the_human_radio | 62 |
| 16 | `syclla_16700gen.npz` | syclla | 16700 |

---

## Section 2: How each model differs

The Original testing for version 2 of Formula Zero lead to the creation of **alpha**, **beta**, **gamma**, **delta**, and **epsilon** models. All of these five models trained the same way and whose purpose was to test the creation of Formula Zero.

With each of these models being trained under the same circumstances, it's easy to think that these would behave the same; however, one cannot be further from the truth. When `Alpha_720gen_v2.npz`, or **Alpha720**, was created, it showed a level of carefulness in which it would take the track at a slower pace, with the upside of being able to navigate trickier circumstances. After seeing this I decided to make the second model, **beta1280**, whose goal was to train longer than the alpha and compare them. As the name suggests, the beta trained for 1280 Generations and showed an interesting, yet predictable outcome. The neural network decided to play things more dangerously and choose to cut corners closer than ever before. Of course this had the problem where it often died in more complicated procedurally generated levels. 

After seeing the impacts of possibly overtraining I decided to make **gamma200**, in which it only trained 200 generations. This model is rather remarkable, it shows that you don't need mass amounts of training to be good. Its playstyle is similar to beta1280 and in simpler courses it shines. Both of the following two models I created with no goal in mind. At the time I made the race mode and wanted more racers. The **delta1050** performs fairly well, achieving similar times and displacing similar strength as beta1280. Then we arrive at **epsilon555**, even though it was trained for 555 generations, it shows a sub-par strength when it comes to the other models, particularly when it comes to the start of the race. This model often crashes and turns too soon, a result I wasn't expecting given the mass amount of training. The alpha stuck out to me, because it showed a level of carefulness that the other models did not. I trained the alpha a further 720 generations (to have 1440 in total). To see how it would adapt to the more training. It seems to be now in a very good spot, where it takes more risk than the Alpha720 version, but not as much risk as the others. This allows it to finish very respectively and makes it beat even the premier models sometimes.

This next section shows the turning point where I tried to make the models smarter, using a variety of methods. The first one was **shadow_fury**, its goal was to see how far the current structure (that is fitness functions, sensors, hidden inputs) could go. After training overnight, it amassed 5252 generations. It showed significant improvement over the previous set of models. With this in mind, I trained it further, now amassing 8452 generations. This model is marginally better than the previous milestone of 5252, which shows that training passes a certain point has negligible effects. 

The next models made was the **valkyrie** class of models. This was a substantial jump in complexity. This model trained with 2 new toggles on, Crossover and Dynamic Mut. Currently, the `create_next_generation` only mutates a single parent. However it is most beneficial if you take the weights from two top-performing cars and combine them. For example, take the first half of the hidden nodes from Parent A, and the second half from Parent B. This allows the network to combine the "good cornering" trait of one car with the "good straightaway speed" trait of another. The second improvement was something called "Dynamic Mut", or dynamic mutation. A fixed `MUTATION_RATE = 0.15` is too high for fine-tuning. Early on, you want high mutation to explore the track. Later, when the cars are almost perfect, large mutations ruin good weights. The last change was model elitism, the idea is keep your top 1-2 cars completely unmutated (Elitism) so you never lose your best progress. But, to prevent the gene pool from stagnating, introduce 5-10% completely random, brand-new networks every generation (Immigrants) to inject fresh DNA. This valkyrie model showed promising results at 1267 generations trained. So, I trained it overnight for a staggering 19767 generations. This improved version showed 40% improvement from the earlier version. 

A problem most models have, including valkyrie, is that they tend to crash into walls on tight turns. To counteract this I made **astraea400** and **astrea1040**, both of which use the same backend as valkyrie, but have a revised reward function. This function added a reward in which the closer you are to the center line, the more fitness you get (with a bonus of up to 4x!). This gave some guidance on the course which allowed astraea to excel. 

The next model, **livid_dagger**, was trained with all the previous advancements in place, but with the change of setting the hidden_nodes to 48 (prior was 24). So far this model has been at par (or sub par sometimes). This can be contributed to the model having more "brain" complexity which hasn't been fully trained yet. When trained to 3500 generations it has shown to have understand the game more. 

As a joke, when demoing formula zero to one of my friends I made a 62 generation model named **the_human_radio** (after his youtube channel). It has shown to be remarkably capable at navigating very hard courses. Achieving the lowest DNF rates by far!

**Syclla_16700** is a model with updated hidden nodes from 48 -> 96. After 16700 generations it has shown State of the art performance in testing. Further test will be done on upping the training and hidden nodes count.

---

## Section 3: Model Stats

### Easy Difficulty
*Baseline for LOS set to: `delta_1050gen_v2`*

| Filename | Elo Rating | Win % | DNF % | Avg Lap (s) | LOS vs Base |
|:---|---:|---:|---:|---:|---:|
| astraea_1040gen | 1813 | 70.9% | 50.9% | 7.152 | 100.0% |
| delta_1050gen_v2 | 1770 | 71.6% | 42.8% | 7.251 | N/A |
| valkyrie_1267gen | 1713 | 68.0% | 42.1% | 7.247 | 0.0% |
| alpha_1440gen_v2 | 1675 | 67.6% | 38.7% | 7.256 | 97.9% |
| valkyrie_19767gen | 1573 | 64.0% | 43.5% | 7.282 | 0.0% |
| astraea_400gen | 1515 | 61.3% | 42.6% | 7.271 | 0.0% |
| livid_dagger_500gen | 1301 | 56.3% | 40.4% | 7.370 | 0.0% |
| syclla_16700gen | 1270 | 58.3% | 40.0% | 7.333 | 0.0% |
| gamma_200gen_v2 | 1188 | 44.8% | 42.0% | 7.398 | 0.0% |
| livid_dagger_3500gen | 1104 | 48.4% | 40.7% | 7.417 | 0.0% |
| the_human_radio_62gen | 1097 | 60.5% | 18.2% | 7.659 | 1.2% |
| shadow_fury_8452gen | 891 | 36.5% | 42.4% | 7.500 | 0.0% |
| beta_1280gen_v2 | 837 | 21.5% | 41.5% | 7.536 | 0.0% |
| shadow_fury_5252gen | 721 | 23.2% | 42.1% | 7.558 | 0.0% |
| alpha_720gen_v2 | 439 | 33.6% | 38.1% | 7.612 | 0.0% |
| epsilon_555gen_v2 | 293 | 11.1% | 45.4% | 7.625 | 0.0% |

### Standard Difficulty
*Baseline for LOS set to: `delta_1050gen_v2`*

| Filename | Elo Rating | Win % | DNF % | Avg Lap (s) | LOS vs Base |
|:---|---:|---:|---:|---:|---:|
| syclla_16700gen | 1429 | 81.0% | 29.1% | 4.938 | 100.0% |
| livid_dagger_3500gen | 1402 | 71.6% | 35.0% | 4.942 | 100.0% |
| astraea_1040gen | 1285 | 51.8% | 60.9% | 4.938 | 53.6% |
| livid_dagger_500gen | 1279 | 51.1% | 44.9% | 5.055 | 98.9% |
| alpha_720gen_v2 | 1262 | 53.5% | 23.9% | 5.418 | 50.0% |
| valkyrie_19767gen | 1261 | 54.7% | 61.7% | 4.924 | 35.4% |
| shadow_fury_8452gen | 1236 | 46.8% | 66.4% | 4.971 | 7.4% |
| the_human_radio_62gen | 1232 | 56.6% | 0.8% | 5.509 | 36.0% |
| delta_1050gen_v2 | 1229 | 48.8% | 46.3% | 5.212 | N/A |
| valkyrie_1267gen | 1227 | 59.7% | 39.0% | 5.027 | 100.0% |
| astraea_400gen | 1152 | 51.9% | 60.5% | 4.949 | 77.3% |
| gamma_200gen_v2 | 1136 | 38.9% | 66.5% | 5.048 | 1.4% |
| alpha_1440gen_v2 | 1117 | 51.5% | 33.2% | 5.302 | 79.3% |
| shadow_fury_5252gen | 1060 | 33.3% | 76.6% | 4.945 | 0.1% |
| beta_1280gen_v2 | 1010 | 32.3% | 70.4% | 5.066 | 0.0% |
| epsilon_555gen_v2 | 883 | 16.1% | 90.6% | 4.938 | 0.0% |

### Hard Difficulty
*Baseline for LOS set to: `delta_1050gen_v2`*

| Filename | Elo Rating | Win % | DNF % | Avg Lap (s) | LOS vs Base |
|:---|---:|---:|---:|---:|---:|
| livid_dagger_3500gen | 1425 | 68.9% | 57.0% | 5.664 | 100.0% |
| syclla_16700gen | 1403 | 73.1% | 48.6% | 5.702 | 100.0% |
| the_human_radio_62gen | 1371 | 74.8% | 4.6% | 6.535 | 100.0% |
| valkyrie_19767gen | 1256 | 53.7% | 78.8% | 5.708 | 24.6% |
| livid_dagger_500gen | 1256 | 58.7% | 63.9% | 5.743 | 100.0% |
| astraea_1040gen | 1211 | 45.2% | 85.0% | 5.729 | 18.3% |
| gamma_200gen_v2 | 1211 | 38.5% | 85.3% | 5.771 | 0.2% |
| alpha_720gen_v2 | 1206 | 59.7% | 47.7% | 6.371 | 97.5% |
| valkyrie_1267gen | 1203 | 56.3% | 68.7% | 5.741 | 99.7% |
| alpha_1440gen_v2 | 1198 | 51.6% | 60.1% | 6.190 | 99.4% |
| astraea_400gen | 1175 | 52.1% | 78.1% | 5.682 | 70.3% |
| shadow_fury_8452gen | 1158 | 47.6% | 77.9% | 5.732 | 20.8% |
| delta_1050gen_v2 | 1139 | 49.1% | 71.1% | 6.032 | N/A |
| shadow_fury_5252gen | 1059 | 27.1% | 90.9% | 5.669 | 0.0% |
| beta_1280gen_v2 | 1016 | 27.0% | 90.8% | 5.808 | 0.0% |
| epsilon_555gen_v2 | 914 | 16.3% | 97.6% | 5.507 | 0.0% |

### Insane Difficulty
*Baseline for LOS set to: `delta_1050gen_v2`*

| Filename | Elo Rating | Win % | DNF % | Avg Lap (s) | LOS vs Base |
|:---|---:|---:|---:|---:|---:|
| the_human_radio_62gen | 1400 | 82.7% | 12.8% | 6.811 | 100.0% |
| alpha_720gen_v2 | 1346 | 65.1% | 64.4% | 6.498 | 100.0% |
| livid_dagger_500gen | 1341 | 59.0% | 77.8% | 5.799 | 100.0% |
| syclla_16700gen | 1340 | 70.8% | 36.1% | 5.010 | 99.9% |
| astraea_400gen | 1292 | 49.1% | 88.9% | 5.689 | 22.6% |
| delta_1050gen_v2 | 1271 | 51.4% | 84.4% | 6.138 | N/A |
| valkyrie_1267gen | 1264 | 54.1% | 84.2% | 5.810 | 38.2% |
| livid_dagger_3500gen | 1237 | 64.0% | 74.7% | 5.694 | 99.1% |
| alpha_1440gen_v2 | 1233 | 51.4% | 75.7% | 6.299 | 65.7% |
| valkyrie_19767gen | 1177 | 34.1% | 66.8% | 4.972 | 97.0% |
| shadow_fury_8452gen | 1173 | 47.1% | 88.5% | 5.814 | 12.6% |
| gamma_200gen_v2 | 1129 | 38.5% | 94.0% | 5.757 | 0.2% |
| astraea_1040gen | 1111 | 49.1% | 92.2% | 5.851 | 5.0% |
| beta_1280gen_v2 | 989 | 23.9% | 96.8% | 5.904 | 0.0% |
| epsilon_555gen_v2 | 969 | 15.5% | 99.9% | 6.183 | 0.0% |
| shadow_fury_5252gen | 928 | 25.4% | 97.3% | 5.750 | 0.0% |

### Extreme Difficulty
*Baseline for LOS set to: `delta_1050gen_v2`*

| Filename | Elo Rating | Win % | DNF % | Avg Lap (s) | LOS vs Base |
|:---|---:|---:|---:|---:|---:|
| the_human_radio_62gen | 1528 | 88.3% | 42.4% | 8.131 | 100.0% |
| alpha_720gen_v2 | 1407 | 69.5% | 91.8% | 7.570 | 99.5% |
| astraea_1040gen | 1280 | 51.6% | 99.0% | 6.779 | 45.0% |
| livid_dagger_3500gen | 1254 | 61.6% | 93.0% | 6.546 | 78.9% |
| syclla_16700gen | 1243 | 60.3% | 91.4% | 6.645 | 97.8% |
| delta_1050gen_v2 | 1236 | 53.2% | 95.8% | 6.959 | N/A |
| valkyrie_1267gen | 1222 | 49.0% | 98.2% | 6.457 | 11.2% |
| gamma_200gen_v2 | 1172 | 41.8% | 98.8% | 6.620 | 17.1% |
| shadow_fury_8452gen | 1162 | 48.4% | 98.5% | 6.530 | 1.8% |
| astraea_400gen | 1162 | 45.7% | 98.5% | 6.375 | 70.7% |
| livid_dagger_500gen | 1159 | 56.3% | 95.0% | 6.571 | 70.4% |
| valkyrie_19767gen | 1133 | 49.9% | 98.5% | 6.498 | 38.2% |
| alpha_1440gen_v2 | 1102 | 48.3% | 95.3% | 7.260 | 73.1% |
| epsilon_555gen_v2 | 1072 | 22.3% | 100.0% | N/A | 0.0% |
| shadow_fury_5252gen | 1046 | 26.7% | 99.4% | 6.427 | 0.0% |
| beta_1280gen_v2 | 1023 | 25.1% | 99.9% | 6.325 | 0.0% |
```
