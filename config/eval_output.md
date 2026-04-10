dwiseman@Darrens-iMac ~ % /Users/dwiseman/github/Motley\ Crews\ Agent\ Battle/scripts/run_eval.command ; exit;
Round-robin: heuristic, random, tuned_a  seeds=0..199

--- heuristic ---

=== heuristic vs random (from focus perspective) ===
  wins=400 losses=0 draws=0 timeouts=0
  games=400  decided(w+l)=400
  win rate (decided): 1.0000  Wilson 95% [0.9905, 1.0000]

=== heuristic vs tuned_a (from focus perspective) ===
  wins=200 losses=200 draws=0 timeouts=0
  games=400  decided(w+l)=400
  win rate (decided): 0.5000  Wilson 95% [0.4512, 0.5488]

--- random ---

=== random vs heuristic (from focus perspective) ===
  wins=0 losses=400 draws=0 timeouts=0
  games=400  decided(w+l)=400
  win rate (decided): 0.0000  Wilson 95% [0.0000, 0.0095]

=== random vs tuned_a (from focus perspective) ===
  wins=0 losses=400 draws=0 timeouts=0
  games=400  decided(w+l)=400
  win rate (decided): 0.0000  Wilson 95% [0.0000, 0.0095]

--- tuned_a ---

=== tuned_a vs heuristic (from focus perspective) ===
  wins=200 losses=200 draws=0 timeouts=0
  games=400  decided(w+l)=400
  win rate (decided): 0.5000  Wilson 95% [0.4512, 0.5488]

=== tuned_a vs random (from focus perspective) ===
  wins=400 losses=0 draws=0 timeouts=0
  games=400  decided(w+l)=400
  win rate (decided): 1.0000  Wilson 95% [0.9905, 1.0000]

=== Elo (from round-robin game outcomes) ===
  tuned_a: 2111.0
  heuristic: 1352.2
  random: 1036.8

(behavior=true not yet aggregated for full round_robin; use pairwise.)

Saving session...
...copying shared history...
...saving history...truncating history files...
...completed.

[Process completed]