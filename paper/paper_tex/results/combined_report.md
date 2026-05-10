# Checkpoint Eval Suite

## Overall

| method | n | success | collision | fill | final dist | p/e/g |
|---|---:|---:|---:|---:|---:|---:|
| dagger | 900 | 0.030 | 0.958 | 0.958 | 1.683 | 0.614/0.409/0.396 |
| fixed | 900 | 0.667 | 0.333 | 0.721 | 0.672 | 0.500/0.500/0.500 |
| flightonly | 900 | 0.751 | 0.249 | 0.975 | 0.550 | 0.577/0.415/0.407 |
| nondiff | 900 | 0.580 | 0.420 | 0.757 | 0.826 | 0.415/0.383/0.507 |
| pretrained | 900 | 0.031 | 0.959 | 0.958 | 1.685 | 0.613/0.406/0.395 |
| randfix | 900 | 0.627 | 0.373 | 0.743 | 0.743 | 0.530/0.515/0.459 |
| zero | 900 | 0.032 | 0.968 | 0.622 | 1.698 | 0.500/0.500/0.500 |

## By Scene

| method | scene | n | success | collision | fill | final dist | p/e/g |
|---|---|---:|---:|---:|---:|---:|---:|
| dagger | dark | 300 | 0.017 | 0.963 | 0.954 | 1.696 | 0.678/0.707/0.695 |
| dagger | glare | 300 | 0.043 | 0.943 | 0.936 | 1.670 | 0.786/0.118/0.132 |
| dagger | specular | 300 | 0.030 | 0.967 | 0.983 | 1.683 | 0.379/0.400/0.361 |
| fixed | dark | 300 | 0.683 | 0.317 | 0.704 | 0.640 | 0.500/0.500/0.500 |
| fixed | glare | 300 | 0.653 | 0.347 | 0.687 | 0.697 | 0.500/0.500/0.500 |
| fixed | specular | 300 | 0.663 | 0.337 | 0.771 | 0.678 | 0.500/0.500/0.500 |
| flightonly | dark | 300 | 0.820 | 0.180 | 0.977 | 0.428 | 0.602/0.607/0.597 |
| flightonly | glare | 300 | 0.670 | 0.330 | 0.958 | 0.697 | 0.704/0.213/0.223 |
| flightonly | specular | 300 | 0.763 | 0.237 | 0.990 | 0.525 | 0.426/0.425/0.401 |
| nondiff | dark | 300 | 0.623 | 0.377 | 0.673 | 0.754 | 0.415/0.383/0.507 |
| nondiff | glare | 300 | 0.600 | 0.400 | 0.665 | 0.795 | 0.415/0.383/0.507 |
| nondiff | specular | 300 | 0.517 | 0.483 | 0.933 | 0.927 | 0.414/0.383/0.507 |
| pretrained | dark | 300 | 0.020 | 0.967 | 0.952 | 1.697 | 0.677/0.703/0.693 |
| pretrained | glare | 300 | 0.037 | 0.950 | 0.940 | 1.680 | 0.781/0.113/0.128 |
| pretrained | specular | 300 | 0.037 | 0.960 | 0.983 | 1.677 | 0.381/0.402/0.365 |
| randfix | dark | 300 | 0.640 | 0.360 | 0.734 | 0.718 | 0.539/0.521/0.453 |
| randfix | glare | 300 | 0.607 | 0.393 | 0.692 | 0.777 | 0.524/0.520/0.466 |
| randfix | specular | 300 | 0.633 | 0.367 | 0.802 | 0.734 | 0.527/0.503/0.457 |
| zero | dark | 300 | 0.030 | 0.970 | 0.585 | 1.699 | 0.500/0.500/0.500 |
| zero | glare | 300 | 0.033 | 0.967 | 0.603 | 1.698 | 0.500/0.500/0.500 |
| zero | specular | 300 | 0.033 | 0.967 | 0.677 | 1.696 | 0.500/0.500/0.500 |

## Camera Phase Means

| method | scene | phase | n | p/e/g | scene effect | clearance |
|---|---|---|---:|---:|---:|---:|
| dagger | dark | after | 330 | 0.505/0.489/0.472 | 0.000 | 0.538 |
| dagger | dark | before | 6869 | 0.681/0.712/0.700 | 0.102 | 0.847 |
| dagger | dark | near | 299 | 0.661/0.706/0.687 | 0.068 | 0.011 |
| dagger | glare | after | 451 | 0.514/0.463/0.454 | 0.000 | 0.595 |
| dagger | glare | before | 8533 | 0.792/0.108/0.123 | 0.100 | 0.828 |
| dagger | glare | near | 319 | 0.750/0.097/0.128 | 0.071 | 0.013 |
| dagger | specular | after | 264 | 0.498/0.471/0.464 | 0.000 | 0.582 |
| dagger | specular | before | 7817 | 0.377/0.398/0.359 | 0.049 | 0.819 |
| dagger | specular | near | 294 | 0.402/0.370/0.378 | 0.045 | 0.008 |
| fixed | dark | after | 6477 | 0.500/0.500/0.500 | 0.000 | 0.708 |
| fixed | dark | before | 7044 | 0.500/0.500/0.500 | 0.387 | 0.865 |
| fixed | dark | near | 1233 | 0.500/0.500/0.500 | 0.134 | 0.023 |
| fixed | glare | after | 6155 | 0.500/0.500/0.500 | 0.000 | 0.717 |
| fixed | glare | before | 7104 | 0.500/0.500/0.500 | 0.511 | 0.865 |
| fixed | glare | near | 1160 | 0.500/0.500/0.500 | 0.496 | 0.023 |
| fixed | specular | after | 6281 | 0.500/0.500/0.500 | 0.000 | 0.708 |
| fixed | specular | before | 7061 | 0.500/0.500/0.500 | 0.183 | 0.864 |
| fixed | specular | near | 1175 | 0.500/0.500/0.500 | 0.062 | 0.023 |
| flightonly | dark | after | 8144 | 0.509/0.484/0.477 | 0.000 | 0.714 |
| flightonly | dark | before | 6633 | 0.682/0.711/0.699 | 0.106 | 0.867 |
| flightonly | dark | near | 1317 | 0.661/0.688/0.677 | 0.025 | 0.025 |
| flightonly | glare | after | 5581 | 0.517/0.448/0.445 | 0.000 | 0.688 |
| flightonly | glare | before | 8057 | 0.795/0.112/0.125 | 0.107 | 0.848 |
| flightonly | glare | near | 1328 | 0.738/0.099/0.127 | 0.045 | 0.025 |
| flightonly | specular | after | 7031 | 0.497/0.468/0.461 | 0.000 | 0.701 |
| flightonly | specular | before | 7250 | 0.377/0.401/0.360 | 0.051 | 0.855 |
| flightonly | specular | near | 1339 | 0.413/0.371/0.387 | 0.015 | 0.026 |
| nondiff | dark | after | 5727 | 0.399/0.352/0.513 | 0.000 | 0.694 |
| nondiff | dark | before | 7250 | 0.423/0.398/0.504 | 0.406 | 0.859 |
| nondiff | dark | near | 1165 | 0.411/0.369/0.508 | 0.148 | 0.023 |
| nondiff | glare | after | 5489 | 0.399/0.353/0.513 | 0.000 | 0.713 |
| nondiff | glare | before | 7316 | 0.423/0.397/0.504 | 0.513 | 0.858 |
| nondiff | glare | near | 1112 | 0.412/0.368/0.508 | 0.493 | 0.023 |
| nondiff | specular | after | 4764 | 0.399/0.352/0.513 | 0.000 | 0.694 |
| nondiff | specular | before | 7280 | 0.420/0.395/0.505 | 0.095 | 0.857 |
| nondiff | specular | near | 1004 | 0.405/0.358/0.512 | 0.033 | 0.022 |
| pretrained | dark | after | 302 | 0.505/0.489/0.472 | 0.000 | 0.549 |
| pretrained | dark | before | 6868 | 0.681/0.707/0.697 | 0.105 | 0.848 |
| pretrained | dark | near | 301 | 0.665/0.705/0.689 | 0.072 | 0.009 |
| pretrained | glare | after | 379 | 0.514/0.462/0.451 | 0.000 | 0.572 |
| pretrained | glare | before | 8564 | 0.786/0.104/0.120 | 0.093 | 0.826 |
| pretrained | glare | near | 311 | 0.725/0.084/0.116 | 0.052 | 0.012 |
| pretrained | specular | after | 319 | 0.499/0.473/0.464 | 0.000 | 0.587 |
| pretrained | specular | before | 7800 | 0.378/0.400/0.363 | 0.050 | 0.820 |
| pretrained | specular | near | 284 | 0.403/0.362/0.375 | 0.042 | 0.010 |
| randfix | dark | after | 6029 | 0.552/0.514/0.441 | 0.000 | 0.703 |
| randfix | dark | before | 7040 | 0.539/0.506/0.454 | 0.328 | 0.865 |
| randfix | dark | near | 1185 | 0.547/0.487/0.449 | 0.117 | 0.023 |
| randfix | glare | after | 5724 | 0.529/0.539/0.476 | 0.000 | 0.715 |
| randfix | glare | before | 7109 | 0.523/0.503/0.461 | 0.473 | 0.865 |
| randfix | glare | near | 1118 | 0.525/0.503/0.456 | 0.466 | 0.022 |
| randfix | specular | after | 5940 | 0.525/0.509/0.459 | 0.000 | 0.704 |
| randfix | specular | before | 7084 | 0.527/0.487/0.456 | 0.225 | 0.865 |
| randfix | specular | near | 1164 | 0.523/0.476/0.455 | 0.080 | 0.024 |
| zero | dark | after | 288 | 0.500/0.500/0.500 | 0.000 | 0.733 |
| zero | dark | before | 6979 | 0.500/0.500/0.500 | 0.342 | 0.876 |
| zero | dark | near | 197 | 0.500/0.500/0.500 | 0.341 | 0.006 |
| zero | glare | after | 317 | 0.500/0.500/0.500 | 0.000 | 0.749 |
| zero | glare | before | 7027 | 0.500/0.500/0.500 | 0.442 | 0.876 |
| zero | glare | near | 178 | 0.500/0.500/0.500 | 0.639 | 0.007 |
| zero | specular | after | 320 | 0.500/0.500/0.500 | 0.000 | 0.671 |
| zero | specular | before | 7006 | 0.500/0.500/0.500 | 0.163 | 0.876 |
| zero | specular | near | 188 | 0.500/0.500/0.500 | 0.166 | 0.006 |

## Diagnostics

- `dagger` glare-vs-dark near camera L1: `0.419` (0.089/0.609/0.560) -> OK.
- `fixed` glare-vs-dark near camera L1: `0.000` (0.000/0.000/0.000) -> weak separation.
- `flightonly` glare-vs-dark near camera L1: `0.405` (0.077/0.589/0.550) -> OK.
- `nondiff` glare-vs-dark near camera L1: `0.000` (0.001/0.000/0.000) -> weak separation.
- `pretrained` glare-vs-dark near camera L1: `0.418` (0.060/0.621/0.573) -> OK.
- `randfix` glare-vs-dark near camera L1: `0.015` (0.023/0.016/0.007) -> weak separation.
- `zero` glare-vs-dark near camera L1: `0.000` (0.000/0.000/0.000) -> weak separation.

Figures are in `figures/`; raw episode and trace CSVs are in `raw/`.
