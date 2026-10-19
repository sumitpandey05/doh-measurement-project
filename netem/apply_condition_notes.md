# Network-condition profiles - how they map across operating systems

Everyone on the team must apply the SAME conditions, or results are not comparable.
Netem (`tc`) is the reference. Other tools only approximate it - say so in the paper.

| Profile      | Linux / WSL2 (`tc netem`, reference)      | macOS: Network Link Conditioner (closest preset)   | Windows native: Clumsy                     |
|--------------|-------------------------------------------|----------------------------------------------------|--------------------------------------------|
| clean        | no rule                                   | Off                                                | stopped                                    |
| moderate     | delay 50ms 10ms, loss 0.5%                | custom: delay 50 ms, packets dropped 0.5%          | Lag 50 ms (out), Drop 0.5% (out)           |
| high_jitter  | delay 80ms 40ms distribution normal       | custom: delay 80 ms (no jitter control -> weaker)  | Lag 80 ms, plus Throttle/Out-of-order      |
| lossy        | delay 30ms 5ms, loss 3%                   | custom: delay 30 ms, packets dropped 3%            | Lag 30 ms (out), Drop 3% (out)             |

Rules of thumb
* netem shapes outgoing packets only: a "50ms" rule adds ~50 ms to the round trip.
  Clumsy: tick "Outbound" only, to match. Network Link Conditioner shapes both ways, so
  halve its numbers to match.
* Record in each CSV's `profile` column the profile NAME, and in your notes which tool made it.
* If any teammate used a non-netem tool, say so in the Methodology and Limitations.
