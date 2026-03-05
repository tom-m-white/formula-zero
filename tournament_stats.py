import csv
import math
import os

def calculate_elo(rating_a, rating_b, score_a, k_factor=32):
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))
    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - score_a) - expected_b)
    return new_rating_a, new_rating_b

def calculate_los(wins, losses):
    if wins + losses == 0:
        return 50.0
    z = (wins - losses) / math.sqrt(2 * (wins + losses))
    los = 0.5 + 0.5 * math.erf(z)
    return los * 100.0

def generate_stats():
    log_file = "tournament_match_log.csv"
    if not os.path.exists(log_file):
        print(f"Error: '{log_file}' not found.")
        print("Please run Tournament Mode in Formula Zero for a few matches first!")
        return


    elo = {}
    matches = {}
    wins = {}
    dnfs = {}
    lap_times = {}
    h2h = {}


    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mod_a = row["Model_A"]
            mod_b = row["Model_B"]
            score_a = float(row["Score_A"])
            stat_a = row["Status_A"]
            stat_b = row["Status_B"]
            lap_a = row["Lap_A"]
            lap_b = row["Lap_B"]

            
            for m in [mod_a, mod_b]:
                if m not in elo:
                    elo[m] = 1200.0
                    matches[m] = 0
                    wins[m] = 0
                    dnfs[m] = 0
                    lap_times[m] = []
                    h2h[m] = {}

            if mod_b not in h2h[mod_a]: h2h[mod_a][mod_b] = [0, 0]
            if mod_a not in h2h[mod_b]: h2h[mod_b][mod_a] = [0, 0]

            matches[mod_a] += 1
            matches[mod_b] += 1

        
            if score_a == 1.0:
                wins[mod_a] += 1
                h2h[mod_a][mod_b][0] += 1
                h2h[mod_b][mod_a][1] += 1
            elif score_a == 0.0:
                wins[mod_b] += 1
                h2h[mod_b][mod_a][0] += 1
                h2h[mod_a][mod_b][1] += 1
            else:
                wins[mod_a] += 0.5
                wins[mod_b] += 0.5

           
            if stat_a != "Finished": dnfs[mod_a] += 1
            if stat_b != "Finished": dnfs[mod_b] += 1

           
            if stat_a == "Finished" and lap_a: lap_times[mod_a].append(float(lap_a))
            if stat_b == "Finished" and lap_b: lap_times[mod_b].append(float(lap_b))

            elo[mod_a], elo[mod_b] = calculate_elo(elo[mod_a], elo[mod_b], score_a)

    baseline = None
    for m in elo.keys():
        if "delta_1050gen_v2" in m:
            baseline = m
            break
    if not baseline:
        for m in elo.keys():
            if "alpha" in m.lower():
                baseline = m
                break

    print("\n" + "="*88)
    print(" FORMULA ZERO - TOURNAMENT STATS ".center(88, "="))
    print("="*88)
    if baseline:
        print(f" Baseline for LOS set to: {baseline.replace('.npz', '')}")
    else:
        print(" No 'alpha' model found. LOS baseline disabled.")
    print("-" * 88)

    header = f"{'Filename':<25} | {'Elo Rating':>10} | {'Win %':>7} | {'DNF %':>7} | {'Avg Lap (s)':>11} | {'LOS vs Base':>11}"
    print(header)
    print("-" * 88)

    sorted_models = sorted(elo.items(), key=lambda x: x[1], reverse=True)

    for mod, rating in sorted_models:
        clean_name = mod.replace('.npz', '')[:25]
        
        win_pct = (wins[mod] / matches[mod] * 100) if matches[mod] > 0 else 0
        dnf_pct = (dnfs[mod] / matches[mod] * 100) if matches[mod] > 0 else 0
        avg_lap = sum(lap_times[mod]) / len(lap_times[mod]) if lap_times[mod] else 0
        
        if baseline and mod != baseline:
            if baseline in h2h[mod]:
                w, l = h2h[mod][baseline]
                los_val = calculate_los(w, l)
            else:
                los_val = 50.0 
            los_str = f"{los_val:>10.1f}%"
        elif mod == baseline:
            los_str = "       N/A " 
        else:
            los_str = "       N/A "

        lap_str = f"{avg_lap:>11.3f}" if avg_lap > 0 else "        N/A"
        
        print(f"{clean_name:<25} | {rating:>10.0f} | {win_pct:>6.1f}% | {dnf_pct:>6.1f}% | {lap_str} | {los_str}")

if __name__ == '__main__':
    generate_stats()