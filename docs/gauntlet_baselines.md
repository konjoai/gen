# Gauntlet baselines (V1 capability profiles)

Reference rows the harness must reproduce: attention aces MQAR across K; the
recurrence shows the known recall gap (degrades as K grows) while still
passing induction and selective-copy. If this separation is absent the harness
is broken (charter V1).

- **gen_version**: 0.1.0
- **python**: 3.11.15
- **platform**: Linux-6.18.5-x86_64-with-glibc2.39
- **torch**: 2.5.1+cpu
- **numpy**: 2.1.3
- **seed**: 0

## Mixer: `attention`

**Verdict:** aces-mqar (recall holds across K)

Environment / versions:

| key | value |
| --- | --- |
| gen_version | 0.1.0 |
| python | 3.11.15 |
| platform | Linux-6.18.5-x86_64-with-glibc2.39 |
| torch | 2.5.1+cpu |
| numpy | 2.1.3 |
| seed | 0 |

### MQAR accuracy-vs-K

| K | accuracy |
| --- | --- |
| 1 | 1.000 |
| 2 | 0.991 |
| 4 | 0.998 |
| 8 | 1.000 |
| 16 | 1.000 |
| 32 | 1.000 |

### Capability profile

| task | accuracy | steps | recurrent step |
| --- | --- | --- | --- |
| induction | 0.998 | 600 | yes |
| mqar_K1 | 1.000 | 300 | yes |
| mqar_K2 | 0.991 | 300 | yes |
| mqar_K4 | 0.998 | 600 | yes |
| mqar_K8 | 1.000 | 1500 | yes |
| mqar_K16 | 1.000 | 600 | yes |
| mqar_K32 | 1.000 | 900 | yes |
| selective_copy | 0.992 | 1200 | yes |
| parity | 0.573 | 3000 | yes |
| mod_add | 1.000 | 300 | yes |
| dyck | 0.541 | 3000 | yes |


## Mixer: `ssm`

**Verdict:** recall-limited (MQAR degrades as K grows)

Environment / versions:

| key | value |
| --- | --- |
| gen_version | 0.1.0 |
| python | 3.11.15 |
| platform | Linux-6.18.5-x86_64-with-glibc2.39 |
| torch | 2.5.1+cpu |
| numpy | 2.1.3 |
| seed | 0 |

### MQAR accuracy-vs-K

| K | accuracy |
| --- | --- |
| 1 | 1.000 |
| 2 | 0.993 |
| 4 | 0.978 |
| 8 | 0.265 |
| 16 | 0.192 |
| 32 | 0.150 |

### Capability profile

| task | accuracy | steps | recurrent step |
| --- | --- | --- | --- |
| induction | 0.993 | 900 | yes |
| mqar_K1 | 1.000 | 300 | yes |
| mqar_K2 | 0.993 | 1500 | yes |
| mqar_K4 | 0.978 | 3000 | yes |
| mqar_K8 | 0.265 | 3000 | yes |
| mqar_K16 | 0.192 | 3000 | yes |
| mqar_K32 | 0.150 | 3000 | yes |
| selective_copy | 0.996 | 900 | yes |
| parity | 0.969 | 3000 | yes |
| mod_add | 1.000 | 300 | yes |
| dyck | 1.000 | 900 | yes |

