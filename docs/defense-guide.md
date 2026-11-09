# Гайд до захисту: Multiple Object Video Detection

Зріз перевірених джерел: `5ec9d1f14a9f9aa9e16a4e56c3113dbf3aad3633`, 9 листопада 2026 року. Гайд охоплює завершені CH-01–CH-13 і CH-15, включно з прийнятим вибором фіналіста та inference-пайплайном. CH-14 належить Назару; його фінальних результатів тут немає. CH-16 ще заблокований. Це матеріал для усного захисту, а не фінальні висновки звіту.

## 1. Що прогнозує проєкт

На вхід надходить кадр аерозйомки VisDrone-VID. Для кожного цільового об’єкта модель повертає **клас, прямокутник і confidence** — оцінку впевненості. Це детекція на окремих кадрах: ми не прогнозуємо траєкторії та не підтримуємо ідентичність об’єкта між кадрами. Confidence не є гарантією правильності рамки чи каліброваною ймовірністю.

| Частина даних | Кадри | Відеопослідовності | Цільові рамки |
|---|---:|---:|---:|
| train | 24 201 | 56 | 1 105 516 |
| validation | 2 846 | 7 | 114 132 |
| test | 3 811 | 48 | Мітки недоступні |

Дані містять зображення, COCO-анотації для train/val, текстові labels/ignore та `test_images.csv`. Train і val розділені за послідовностями, без спільних кліпів: сусідні кадри не можна випадково розкидати між ними. Приховані test-мітки потрібні Kaggle для оцінювання.

Джерела: [структура й перевірені факти](../results/ch-02-protocol/data_facts.json), [EDA](../results/ch-06-eda/key_numbers.json), [завантаження даних](../movd/data.py).

## 2. Класи й три формати

| Клас | COCO category_id | YOLO / submission class |
|---|---:|---:|
| person | 1 | 0 |
| bicycle | 2 | 1 |
| car | 3 | 2 |
| van | 4 | 3 |
| truck | 5 | 4 |
| tricycle | 6 | 5 |
| bus | 7 | 6 |
| motor | 8 | 7 |

Person об’єднує початкові VisDrone pedestrian/people, tricycle — дві початкові категорії трициклів. Не слід повторно використовувати початкові VisDrone ID як наші класи.

- **COCO ground truth:** JSON з images, annotations, categories; `bbox=[x,y,w,h]` у пікселях оригіналу, `(x,y)` — лівий верхній кут. Є ознака `iscrowd`.
- **COCO predictions:** список `{image_id, category_id, bbox, score}`. Тут category_id — **1–8**, а рамка також у вихідних пікселях. Це спільний формат усіх моделей перед evaluator.
- **YOLO labels:** рядок `class cx cy w h`, де class — **0–7**, а центр, ширина й висота нормовані на ширину/висоту зображення. Confidence у навчальній мітці немає. У наших ignore-файлах — лише нормовані `cx cy w h`, без класу.
- **Submission CSV:** заголовок `image_id,PredictionString`; один рядок на кожен test-кадр у порядку `test_images.csv`. Усередині PredictionString повторюються групи `class score x y width height`: класи **0–7**, координати лівого верхнього кута й розміри — **пікселі оригіналу**, а не YOLO-нормалізація. Порожній прогноз допустимий.

Перехід класів: `submission_class = category_id - 1`. Test image_id у COCO predictions — числовий порядковий ID рядка, а CSV використовує рядковий ключ кадру з `test_images.csv`. Повернення рамок після resize у вихідні координати обов’язкове.

Джерела: [frozen label map](../configs/protocol.json), [YOLO adapter](../movd/ultra.py), [submission writer і validator](../movd/submission.py).

## 3. Ignored/crowd regions

**Training.** Crowd-рамки не стають цільовими об’єктами. Ігноровані пікселі заповнюються RGB `(124,116,104)` — середнім ImageNet, але пікселі всередині справжніх цільових рамок зберігаються. Так ми зменшуємо видимий неанотований фон, не стираючи цілі в зоні перекриття. Це евристика заповнення, а не пряме вимкнення loss для всіх таких областей; фонова складова loss може залишатися.

**Evaluation.** Використовуємо оригінальний COCO GT без видалення crowd-анотацій. Одна ignore-область повторена для кожної з восьми категорій; це не вісім реальних об’єктів. Невідповідний звичайній цілі прогноз, який достатньо перекриває crowd-область, може бути ignored: він не TP і не FP. Для crowd pycocotools використовує перетин, поділений на площу **прогнозу**, а не звичайне IoU двох рамок. Поріг залежить від поточного evaluation IoU.

Навіщо це перевіряли: GT як predictions дає val_full mAP **1.0000**. Додавання прогнозів в ignore-зони на val_fast зберігає **1.0000**; видалення crowd із GT знижує контроль до **0.8667**. Отже, «просто викинути crowd» змінює оцінювання.

Джерела: [маскування](../movd/data.py), [evaluator](../movd/evaluate.py), [контрольні результати](../results/ch-03-evaluator/sanity.json), [візуальна перевірка](../results/ch-02-protocol/figures/ignore_masking_check.png).

## 4. Навіщо три списки кадрів

| Список | Обсяг | Призначення |
|---|---|---|
| train_rubric | 2 800 кадрів: 50 рівномірно з кожного train-кліпу | Однакова навчальна вибірка для порівняння сімейств |
| val_fast | 714 кадрів: кожен четвертий у кожному val-кліпі | Криві за епохами та вибір варіанта за зафіксованим правилом |
| val_full | Усі 2 846 val-кадрів | Підсумкові AP, які наводимо в таблицях |

Для включених rubric-запусків (CH-10 imgsz1600 — окремий виняток): **шість епох = 16 800 побачених зображень**, seed **42**, базовий розмір **1344**, AMP і горизонтальний flip із імовірністю **0.5**. Torchvision запускався з batch **2**, YOLO11s/RT-DETR — з batch **1**; однакова кількість зображень не означає однакові optimizer steps, FLOPs чи час.

Torchvision обмежує довгу сторону, YOLO застосовує letterbox зі збереженням пропорцій; RT-DETR у реалізованому adapter розтягує кадр до квадрата **1344×1344**. Нормалізація теж модельна: ImageNet mean/std у torchvision, масштабування до **0–1** в Ultralytics. Це важливі межі «однакового протоколу». Старі v0, smoke і CPU-перевірки не змішуємо з повними порівняннями. Для Faster R-CNN/RetinaNet вибирали варіант за final-epoch val_fast; YOLO baseline заздалегідь оцінює фінальний checkpoint, а не найкращу постфактум епоху.

Джерела: [протокол і списки](../configs/protocol.json), [Faster R-CNN protocol](../results/ch-07-frcnn/protocol.md), [YOLO baseline](../results/yolo_baseline_marko_3050/README.md), [геометрія RT-DETR](../movd/rtdetr.py).

## 5. Як читати метрики

**IoU** — площа перетину рамок, поділена на площу їх об’єднання. Для звичайної цілі правильний клас і достатнє IoU потрібні одночасно. Precision показує частку правильних серед прогнозів, recall — частку знайдених цілей. AP узагальнює precision–recall при зміні порогу confidence; mAP усереднює AP за класами й заданими порогами IoU.

- **COCO mAP:** AP на IoU **0.50:0.05:0.95**. Це основний показник; **0.2577** — частка, не 0.2577 відсотка.
- **AP50 / AP75:** пороги IoU **0.50 / 0.75**; другий вимагає точнішої локалізації.
- **AP_S / AP_M / AP_L:** стандартні COCO-діапазони площі в **оригінальних** пікселях, з межами **32² і 96²**. Це не розмір після resize і не довжина найкоротшої сторони.
- **500 detections:** спочатку evaluator залишає top-500 прогнозів **на кадр сумарно за всіма класами** за score. Самого maxDets pycocotools недостатньо, бо він рахує окремо за класами. Для RT-DETR ефективний ліміт додатково обмежений кількістю queries.

Ліміт **100** зберігається лише як окремий legacy-контроль. Внутрішні метрики тренувального фреймворку не замінюють наш evaluator. Ідентичність прихованому Kaggle scorer не доведена. Джерело: [movd/evaluate.py](../movd/evaluate.py).

## 6. Що завершено

| Задача | Що зроблено й де шукати доказ |
|---|---|
| CH-01 | Перевірка структури/повноти датасету, середовища, CUDA та Git-ignore; [check_dataset](../scripts/check_dataset.py), [pins](../requirements.txt) |
| CH-02 | Frozen splits, class map, бюджет, ignore policy; [protocol](../configs/protocol.json), [data facts](../results/ch-02-protocol/data_facts.json) |
| CH-03 | Спільний COCO evaluator, global cap, speed benchmark і sanity controls; [sanity](../results/ch-03-evaluator/sanity.json), [speed self-test](../results/ch-03-evaluator/speed_selftest.json) |
| CH-04 | Torchvision training core з AMP, per-epoch saving/resume та evaluation; [trainer](../movd/tv_train.py), [тести](../tests/test_tv_train.py) |
| CH-05 | Test inference, writer/validator, round-trip та прийнята перша submission; [receipt](../results/ch-05-submission/frcnn_rpn_pre3000_e6_test/validation.txt), [log](../results/submissions.csv) |
| CH-06 | Виправлена EDA: класи, розміри, щільність, послідовності, occlusion, redundancy; [key numbers](../results/ch-06-eda/key_numbers.json) |
| CH-07 | Faster R-CNN baseline і два окремі варіанти, метрики/криві/speed, приклади помилок; [protocol](../results/ch-07-frcnn/protocol.md), [failure analysis](../results/ch-07-frcnn/failure_analysis.md) |
| CH-08 | RetinaNet baseline, smaller anchors і focal gamma; [порівняння](../results/ch-08-retina/results.md) |
| CH-09 | Маскований YOLO export, coordinate/class adapter, повний YOLO11s baseline на ноутбуці Марка, notebook evidence і перевірена передача артефактів; [baseline](../results/yolo_baseline_marko_3050/README.md), [закритий PR](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/10) |
| CH-10 | YOLO baseline replicate, P2 head після виправленого smoke та imgsz1600 exception; [аналіз](../results/ch-10-yolo-tuning/analysis.md) |
| CH-11 | RT-DETR baseline, queries/LR experiments, memory/resume checks і learning curves; [аналіз](../results/ch-11-rtdetr/analysis.md) |
| CH-12 | Cross-family comparison, спільний speed benchmark та RT-DETR-l finalist; [рішення](../results/ch-12-comparison/decision.md) |
| CH-13 | Inference search без training: flip/tiling та їхня комбінація, негативні варіанти, frozen settings; [аналіз](../results/ch-13-inference/analysis.md) |
| CH-15 | Report scaffold, M1 і перевірка GitHub Pages; [M1 PR](https://github.com/Marko-Zenon/Multiple-Object-Video-Detection/pull/9) |

Завершеність цих задач підтверджує [board і evidence logs](../plan.md). Цей гайд не змінює їхніх станів.

## 7. Архітектури: як пояснити без коду

**Faster R-CNN.** Backbone ResNet-50-FPN v2 створює ознаки кількох масштабів. RPN пропонує області-кандидати; другий етап працює з RoI, класифікує їх і уточнює рамки. NMS прибирає надлишкові перекриті прогнози. Два етапи дають окремий контроль пропозицій і foreground/background sampling, але мають обчислювальну вартість. [Builder/trainer](../movd/tv_train.py), [протокол](../results/ch-07-frcnn/protocol.md).

**RetinaNet.** ResNet-50-FPN v2 одразу прогнозує класи й поправки до anchors на кількох масштабах, без RPN і RoI-етапу. Багато anchors потрапляють на простий фон. Focal loss зменшує внесок легких прикладів множником `(1-p_t)^gamma`; alpha задає баланс позитивної/негативної складових. Gamma не є learning rate. Менша gamma послаблює це пригнічення; вона може дати більше корисного сигналу, але й більше впливу фону. [Налаштування й мотивація](../results/ch-08-retina/preflight.md).

**YOLO11s.** Компактна anchor-free модель: CNN-ознаки різних масштабів використовуються для прогнозу класу й рамки без набору наперед заданих anchor-прямокутників. Відсутність anchors не означає відсутність правил призначення позитивних позицій під час навчання. У цьому проєкті використовуємо штатне навчання Ultralytics, не вводимо власний assigner. При inference цей YOLO використовує NMS; це не NMS-free варіант. Export маскує ignore, adapter відновлює COCO-класи та координати. [Реалізація інтеграції](../movd/ultra.py), [фактична конфігурація](../results/yolo_baseline_marko_3050/config.json).

**RT-DETR-l.** CNN-ознаки проходять через encoder/decoder; decoder queries — набір кандидатів, кожен із яких формує прогноз рамки та класу. Bipartite matching під час training узгоджує прогнози з GT і не стимулює кілька позитивних прогнозів однієї цілі. У використаній реалізації inference не потребує NMS. Baseline має **300** queries, варіант — **500**. Більше queries не гарантує кращу якість: змінюється і навчання, а не лише вихідний cap. [RT-DETR adapter](../movd/rtdetr.py), [експериментальні пояснення](../results/ch-11-rtdetr/analysis.md).

## 8. Що змінювали і чому

| Сімейство | Окрема зміна | Гіпотеза та виміряний результат |
|---|---|---|
| Faster R-CNN | Training RPN pre-NMS: 2000 → 3000 | Більше кандидатів для дрібних цілей. Невеликий приріст mAP/AP_S, падіння AP_L і throughput; обрано серед варіантів сімейства |
| Faster R-CNN | RoI positive fraction: 0.25 → 0.50 | Частіше навчатися на foreground. Загальний mAP нижчий за baseline |
| RetinaNet | Менші anchor scales, найдрібніша група [24,30,38] замість default; ratios незмінні | Розміри спираються на train-статистику після resize. Загальний AP зріс, AP_S не покращився |
| RetinaNet | Focal gamma: 2.0 → 1.5, alpha 0.25 | Перевірити користь легших прикладів. Загальний AP впав, AP_S трохи зріс |
| YOLO11s, CH-09 | Baseline без tuning; batch 1, workers 0, AMP | Запустити весь frozen budget на RTX 3050 без скорочення resolution чи епох; це адаптація ресурсів, не доказ приросту якості |
| RT-DETR | Queries: 300 → 500 | Більше кандидатів для щільних сцен. mAP і AP_S впали; гіпотеза не підтвердилася |
| RT-DETR | lr0: 0.000833 → 0.0001, bias warmup 0.0 збережено | Окрема перевірка нижчого LR. Виправлений запуск також гірший за baseline |
| RT-DETR, confounded | lr0 0.0001 разом із bias warmup 0.1 | Дві зміни одночасно. Результат збережено, але ним не можна ізольовано доводити ефект LR |

CH-10 додав два контрольовані варіанти: **P2 stride 4 при 1344** і **imgsz1600** як заздалегідь дозволений resolution exception. Baseline на RTX 5070 є повтором для оцінки run/hardware spread, а не новою архітектурою. Перший P2 smoke мав помилку перенесення pretrained class rows через placeholder names; його збережено, а повний запуск використовує виправлений init. Final-epoch val_fast правило вибрало **yolo_imgsz1600**, а найкращий при frozen 1344 — **yolo_p2**. P2 підвищив загальний AP, але знизив AP_small: гіпотеза користі high-resolution head для малих цілей тут не підтвердилась. [Selection](../results/ch-10-yolo-tuning/selection.json), [smoke/transfer evidence](../results/ch-10-yolo-tuning/analysis.md).

Джерела: [Faster R-CNN protocol](../results/ch-07-frcnn/protocol.md), [RetinaNet preflight](../results/ch-08-retina/preflight.md), [YOLO README](../results/yolo_baseline_marko_3050/README.md), [RT-DETR analysis](../results/ch-11-rtdetr/analysis.md), [помилка warmup](../results/rtdetr_lr1e4/README.md). Негативний результат — теж відповідь на заздалегідь сформульовану гіпотезу; причину падіння не слід видавати за експериментально доведену.

## 9. Перевірені результати

У таблиці — 14 повних rubric-запусків CH-07/08/09/10/11, не smoke/v0. Imgsz1600 — окремий resolution exception; confounded LR-run не входить до primary comparison. Кожне посилання веде до committed `metrics.json`; поля `val_full.metrics` для torchvision, `metrics` для Ultralytics. Усі headline-оцінки: **val_full, 2846 кадрів, global maxDets 500**. Значення округлені до чотирьох знаків.

| Запуск | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| [Faster R-CNN baseline](../results/frcnn_baseline/metrics.json) | 0.1873 | 0.3808 | 0.1604 | 0.0913 | 0.1779 | 0.3452 |
| [Faster R-CNN RPN pre3000, обраний](../results/frcnn_rpn_pre3000/metrics.json) | 0.1893 | 0.3847 | 0.1614 | 0.1083 | 0.1868 | 0.3107 |
| [Faster R-CNN RoI pos50](../results/frcnn_roi_pos50/metrics.json) | 0.1814 | 0.3643 | 0.1553 | 0.1004 | 0.1763 | 0.3109 |
| [RetinaNet baseline](../results/retina_baseline/metrics.json) | 0.1623 | 0.3137 | 0.1458 | 0.0780 | 0.1492 | 0.3045 |
| [RetinaNet anchors, обраний](../results/retina_anchors/metrics.json) | 0.1693 | 0.3465 | 0.1446 | 0.0754 | 0.1528 | 0.3223 |
| [RetinaNet focal gamma 1.5](../results/retina_focal_gamma15/metrics.json) | 0.1524 | 0.3006 | 0.1362 | 0.0832 | 0.1432 | 0.3002 |
| [YOLO11s baseline](../results/yolo_baseline_marko_3050/metrics.json) | 0.1439 | 0.2925 | 0.1192 | 0.0866 | 0.1434 | 0.2555 |
| [YOLO11s baseline replicate RTX 5070](../results/yolo_baseline_5070/metrics.json) | 0.1502 | 0.3079 | 0.1252 | 0.0949 | 0.1520 | 0.2542 |
| [YOLO11s P2, найкращий при 1344](../results/yolo_p2/metrics.json) | 0.1626 | 0.3063 | 0.1510 | 0.0790 | 0.1483 | 0.3124 |
| [YOLO11s imgsz1600, exception / CH-10 selection](../results/yolo_imgsz1600/metrics.json) | 0.1710 | 0.3397 | 0.1482 | 0.0894 | 0.1760 | 0.2748 |
| [RT-DETR-l baseline](../results/rtdetr_baseline/metrics.json) | 0.2577 | 0.4775 | 0.2392 | 0.1296 | 0.2246 | 0.4538 |
| [RT-DETR-l 500 queries](../results/rtdetr_q500/metrics.json) | 0.2389 | 0.4556 | 0.2145 | 0.1281 | 0.2140 | 0.4027 |
| [RT-DETR-l corrected LR 1e-4](../results/rtdetr_lr1e4_wb0/metrics.json) | 0.2179 | 0.4261 | 0.1956 | 0.1015 | 0.2120 | 0.3693 |
| [RT-DETR-l LR + warmup, confounded](../results/rtdetr_lr1e4/metrics.json) | 0.1560 | 0.3066 | 0.1412 | 0.0639 | 0.1556 | 0.2818 |

Позначка «обраний» у цій таблиці стосується тільки експериментів сімейства. Confounded-запуск є завершеним вимірюванням, але не коректним однофакторним tuning-порівнянням.

### Швидкість: три різні контексти

**Torchvision, RTX 5070 Laptop, окремі сесії:** batch one, GPU-resident tensors, включено model transform і NMS; decoding та CPU→GPU transfer виключені.

| Запуск | FPS | мс/кадр |
|---|---:|---:|
| [Faster R-CNN baseline](../results/frcnn_baseline/speed.json) | 18.69 | 53.50 |
| [Faster R-CNN RPN pre3000, обраний](../results/frcnn_rpn_pre3000/speed.json) | 14.85 | 67.35 |
| [Faster R-CNN RoI pos50](../results/frcnn_roi_pos50/speed.json) | 17.48 | 57.19 |
| [RetinaNet baseline](../results/retina_baseline/speed.json) | 12.67 | 78.94 |
| [RetinaNet anchors, обраний](../results/retina_anchors/speed.json) | 19.71 | 50.73 |
| [RetinaNet focal gamma 1.5](../results/retina_focal_gamma15/speed.json) | 24.22 | 41.29 |

**YOLO11s на RTX 3050 Laptop Марка, FP32:** **36.37 FPS / 27.49 мс на кадр**. Preloaded BGR frames, включено letterbox/normalization, transfer і NMS, виключено disk I/O. [Оригінальний benchmark](../results/yolo_baseline_marko_3050/speed.json).

**Ultralytics, одна спільна сесія на RTX 5070 Laptop:** однаковий benchmark із preloaded BGR frames, CPU preprocessing і transfer; **10 warmup + 30 timed frames**, batch **1**. RT-DETR має square resize і не має NMS, YOLO — letterbox та NMS.

| Checkpoint | FPS | мс/кадр |
|---|---:|---:|
| [YOLO11s baseline](../results/ch-11-rtdetr/speed_recheck/yolo_baseline_marko_3050.json) | 64.21 | 15.57 |
| [RT-DETR-l baseline](../results/ch-11-rtdetr/speed_recheck/rtdetr_baseline.json) | 10.60 | 94.31 |
| [RT-DETR-l 500 queries](../results/ch-11-rtdetr/speed_recheck/rtdetr_q500.json) | 10.50 | 95.22 |
| [RT-DETR-l corrected LR 1e-4](../results/ch-11-rtdetr/speed_recheck/rtdetr_lr1e4_wb0.json) | 10.69 | 93.52 |
| [RT-DETR-l LR + warmup, confounded](../results/ch-11-rtdetr/speed_recheck/rtdetr_lr1e4.json) | 10.61 | 94.24 |

FPS між GPU або різними timing scopes **не є прямим порівнянням архітектур**. Джерело методики: [same-session recheck](../results/ch-11-rtdetr/speed_recheck/README.md). Це throughput без дискового читання, не end-to-end відеосистема.

### CH-12: спільна методика для всіх сімейств

Попередні таблиці — історичні сесії. Для фінального порівняння CH-12 виміряв сімейства в одній RTX 5070-сесії: FP32, preloaded BGR host arrays, CPU preprocessing/transfer, resize, inference і postprocessing, без disk I/O; медіани трьох interleaved rounds. Тому не треба ранжувати моделі за старими native torchvision FPS.

| Запуск | Common-scope FPS, RTX 5070 | мс/кадр |
|---|---:|---:|
| frcnn_rpn_pre3000 | 9.10 | 109.84 |
| retina_anchors | 6.29 | 158.92 |
| yolo_baseline_marko_3050 | 75.24 | 13.29 |
| yolo_p2 | 55.41 | 18.05 |
| yolo_imgsz1600 | 49.67 | 20.13 |
| rtdetr_baseline | 13.92 | 71.85 |

Джерело: [CH-12 summary з checkpoint SHA-256 і scopes](../results/ch-12-comparison/speed_same_session/summary.json). Відмінності resize геометрії залишаються навіть у спільному timing scope.

## 10. Як пояснити перевагу RT-DETR і швидкість YOLO

У межах прийнятого CH-12 порівняння RT-DETR-l baseline має найвищий val_full mAP і кращі size-bucket AP. Його val_fast крива вища за кожну CNN-криву на відповідних епохах, починаючи з першої; максимум — на третій. YOLO baseline досягає максимуму val_fast на другій епосі, після чого AP зменшується попри спад training losses. Це спостереження, а не доказ конкретної причини перенавчання. [Криві](../results/ch-11-rtdetr/curves.csv).

Можливі пояснення RT-DETR — pretrained ознаки, encoder/decoder та matching; окремого ablation, який доводить внесок кожного, немає. Не можна робити висновок, що будь-який transformer завжди кращий або швидше збігається. Також budget рівний за images seen, а не обчисленнями, і resize геометрія різна.

YOLO11s у спільній RTX 5070 сесії набагато швидший за RT-DETR-l, але менш точний. Компактність моделі узгоджується з цим: у CH-11 зафіксовано **9.4 млн** параметрів YOLO11s проти **32.8 млн** RT-DETR-l. Це не ізольований доказ причини різниці FPS. Вислів «YOLO найшвидший» тут означає включений same-session Ultralytics benchmark; не доводимо глобальний рейтинг усіх моделей на різних GPU. [Model-size і speed evidence](../results/ch-11-rtdetr/analysis.md).

**CH-12 вибрав RT-DETR-l як фіналіста.** Lead над Faster R-CNN — **0.0683 mAP**, приблизно **10.8×** spread одного baseline replicate pair; виграно **27/27** вторинних перевірок за AP і density bins. Це не статистичний тест. Для fixed-resolution YOLO представником є P2, а imgsz1600 виключено з primary comparison разом із confounded LR-run. [Selection record](../results/ch-12-comparison/selection.json).

Frozen [final.json](../configs/runs/final.json) планує **6072** train-кадри (кожен четвертий), **6** епох = **36432** images seen, imgsz **1344**, queries **300**, решта baseline settings незмінна. Це конфігурація майбутньої/поточної роботи Назара, не завершений final-training result. Fallback — rubric RT-DETR baseline; остаточні checkpoint/submission selection потребують CH-14.

## 11. Проблеми даних і наслідки

- **Малі цілі:** у train **24.91%** цілей мають COCO small-площу в оригіналі; після frozen resize **8.26%** нижче еквівалентного розміру 16 px (`sqrt(area)`). Тому важливі високий input size, AP_S та приклади локалізації; «об’єкт видно» ще не означає достатнє IoU.
- **Щільність:** максимум **180** цілей у train-кадрі й **116** у val. Cap 100 може обмежувати recall; cap 500 і density-bin оцінювання це роблять видимим. Але різні bins мають різний склад класів/сцен, тому вищий AP у щільному bin не доводить, що щільність допомагає.
- **Imbalance:** car — **45.71%** train-цілей, person — **44.08%** val-цілей; bus має лише **264** val-рамки. Дивимось per-class AP; aggregate не описує всі класи однаково.
- **Occlusion:** у val **52.41%** цілей частково й **6.64%** сильно перекриті. Це ускладнює class cues, локалізацію і розділення сусідів.
- **Resolution/scene shift:** розміри кадрів різняться за послідовностями; test містить і менші роздільності, яких немає в train. Однаковий resize не робить візуальний масштаб однаковим.
- **Кореляція:** сусідні train-кадри повторюють **99.5%** track-цілей. train_rubric зберігає **98.6%** train-треків при median gap **8** кадрів, але не гарантує репрезентативність кожної рідкісної події.

Джерела: [усі наведені EDA-числа](../results/ch-06-eda/key_numbers.json), [за послідовностями](../results/ch-06-eda/tables/per_sequence.csv), [помилки на конкретних кадрах](../results/ch-07-frcnn/failure_analysis.md). Один seed на конфігурацію та один baseline replicate на іншому GPU не дають статистичної значущості. CH-10 зафіксував spread **0.0063 mAP на val_full** і **0.0092 на val_fast**; це суміш hardware/nondeterminism, не довірчий інтервал. [Джерело spread](../results/ch-10-yolo-tuning/selection.json). Ілюстрації помилок не є частотним аналізом усіх помилок.

## 12. Перша Kaggle submission

Надіслано Faster R-CNN RPN pre3000, фінальна епоха: внутрішній val_full mAP **0.1893354654**, public score **0.2011538266**, статус **accepted**. Competition baseline у committed upload log — **0.2216324580**: наша перша submission нижча за нього.

Це підтвердило працездатність inference→CSV→validator→upload, але не виконало вимогу перевершити baseline й не визначило фінальну модель. Валідатор перевірив **3811** рядків і **261940** detections; lossless round-trip не змінює predictions. Внутрішній AP і Kaggle score не потрібно прирівнювати: інший split і прихований scorer. Фінальні **дві** submission мають обиратися насамперед за internal validation.

Джерела: [upload log із точним baseline](../results/submissions.csv), [validator receipt](../results/ch-05-submission/frcnn_rpn_pre3000_e6_test/validation.txt), [round-trip](../results/ch-05-submission/roundtrip_frcnn_rpn_pre3000_e6_lossless.json), [правило selection](../configs/protocol.json).

### CH-13: що покращило inference

На одному незміненому rubric RT-DETR checkpoint перевірили **19** predeclared конфігурацій (reference та **18** змін) і потім **2** записані комбінації. Screening був на val_fast; val_full — лише для reference, двох прийнятих окремих змін та обраного stack. Збільшення input size, WBF/ensemble, confidence/top-k і post-hoc NMS дали негативний, нейтральний або недостатній ефект за правилом. Це відхилення від початкового scope «все на val_full» явно зафіксовано. [Повні результати й правило](../results/ch-13-inference/analysis.md).

| Inference variant | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| [ref](../results/ch-13-inference/runs/val_full/ref/metrics.json) | 0.2577 | 0.4775 | 0.2392 | 0.1296 | 0.2246 | 0.4538 |
| [flip_nms06](../results/ch-13-inference/runs/val_full/flip_nms06/metrics.json) | 0.2660 | 0.4941 | 0.2472 | 0.1345 | 0.2302 | 0.4622 |
| [tile_nms05](../results/ch-13-inference/runs/val_full/tile_nms05/metrics.json) | 0.2644 | 0.4991 | 0.2384 | 0.1425 | 0.2367 | 0.4404 |
| [combo_flip_tile_nms05](../results/ch-13-inference/runs/val_full/combo_flip_tile_nms05/metrics.json) | 0.2696 | 0.5069 | 0.2453 | 0.1467 | 0.2375 | 0.4502 |

Обрано **flip + tiling**: full frame, flip і чотири overlapping tiles, разом **6 passes при 1344**; class-wise merge NMS IoU **0.5**, top **500** після merge. Хоча базовий RT-DETR не має NMS, багатовидові прогнози треба звести: це інший, pipeline-level NMS. [Frozen inference config](../configs/runs/final_inference.json).

На RTX 5070 FP32, у спільній CH-13-сесії: reference **13.68 FPS / 73.11 мс**, stack **2.26 FPS / 442.41 мс**; час охоплює passes і CPU decoding/merge, але не disk I/O. Оцінка для test — **28.1 хв**, не факт виконаного test inference. AP_large падає **0.4538 → 0.4502**, densest bin **0.3679 → 0.3614**; top500 binds на **2840 із 2846** кадрів. Тому «все покращилося» — неправильний висновок. [Speed](../results/ch-13-inference/speed/combo_flip_tile_nms05.json), [comparison](../results/ch-13-inference/comparison.csv).

Round-trip raw predictions не пройшов початковий strict tolerance через clipping позакадрових рамок writer-ом. Після однакового clipping обох шляхів різниця точно **0.0**; це пояснена розбіжність, а не прихований PASS. [Перевірка](../results/ch-13-inference/roundtrip_clipping_check.json). CH-13 не тренував модель і не робив Kaggle upload; виграш на rubric checkpoint треба перевірити на final checkpoint у CH-14.

## 13. Імовірні питання на захисті

1. **Що повертає модель?** Клас, confidence і рамку для кожної знайденої цілі на окремому кадрі; не track ID.
2. **Чому split за відео, а не кадрами?** Сусідні кадри дуже схожі. Frame-level split створював би витік сцени й об’єктів у validation.
3. **Чому не навчали всі кадри в кожному порівнянні?** Фіксована рівномірна вибірка з кожного кліпу скорочує повторення й дає здійсненний спільний бюджет. Її обмеження показані в EDA.
4. **Чим val_fast відрізняється від val_full?** Перший потрібен для дешевших кривих і зафіксованого variant selection, другий — для всіх headline-оцінок.
5. **Чому COCO class ID не можна напряму писати в submission?** У нашому COCO це 1–8, у submission/YOLO — 0–7; потрібен узгоджений зсув, а не зміна назв класів.
6. **Яка різниця координат YOLO і CSV?** YOLO labels — нормований центр і розміри; CSV — піксельний лівий верхній кут і розміри оригіналу, ще й confidence.
7. **Чому crowd не просто видалили?** Evaluator має знати, де прогноз не треба штрафувати. Crowd не є звичайною навчальною ціллю, але має залишатися evaluation ignore.
8. **Чи маскування вимикає background loss?** Ні. Воно прибирає візуальний вміст області, але не реалізує точне per-pixel/per-anchor виключення loss.
9. **Чим AP50 відрізняється від mAP?** AP50 перевіряє один поріг overlap, COCO mAP усереднює кілька порогів і суворіше враховує локалізацію.
10. **Чому AP_small часто гірший?** У малої цілі мало пікселів; невеликий зсув рамки сильно змінює IoU, особливо після resize та при occlusion.
11. **Чому top-500 застосовуємо самі?** Потрібний спільний cap на кадр за всіма класами; стандартний per-class maxDets цього не гарантує.
12. **Яка роль RPN і RoI у Faster R-CNN?** RPN шукає кандидати, RoI-етап уточнює їхні класи й рамки. Тому proposal budget і foreground sampling — різні контрольовані зміни.
13. **Навіщо focal loss?** Зменшити домінування численних легких негативів у dense anchor-класифікації. Менша gamma не гарантує вищий AP — це перевіряється експериментом.
14. **Що означає anchor-free у YOLO11s?** Немає наперед заданих anchor-прямокутників, але залишаються просторові позиції ознак, assignment під час training і NMS при inference.
15. **Чому RT-DETR не використовує NMS?** Set prediction і matching навчають прогнози відповідати цілям без звичайного набору дубльованих positive anchors; наш predictor декодує queries без NMS.
16. **Чому 500 queries не допомогли?** У цьому бюджеті AP впав. Queries змінюють matching і decoder training; точну причину не ізолювали, тому «більше місць для прогнозів» не є гарантією.
17. **Чому старий LR-run не є чесним tuning?** Разом із LR змінився bias warmup. Для висновку про LR використовуємо виправлений запуск зі збереженим warmup.
18. **Чи можна сказати, що YOLO швидший лише за цифрами 3050/5070?** Ні. Порівнюємо його з RT-DETR у спільній 5070-сесії; різні пристрої й timing scopes пояснюємо окремо.
19. **Яку модель та inference обрано?** CH-12 вибрав RT-DETR-l, CH-13 — flip + tiles зі спільним NMS. Це не означає завершення final training чи успіху Kaggle: їхні докази має надати Назар у CH-14.
20. **Що доводить Kaggle score і чи невдалі експерименти зайві?** Перша submission довела працездатність формату, але baseline не перевершила. Негативні експерименти потрібні для перевірки гіпотез; їх не видаляємо й не пояснюємо неперевіреними причинними твердженнями.

## 14. Що ще не завершено

CH-10, CH-12 і CH-13 завершені й інтегровані; вони більше не pending. CH-09–CH-13 і CH-15 мають DONE на перевіреному [board](../plan.md). CH-14 зарезервований за Назаром прямим погодженням команди; його READY-рядок у цьому зрізі не є дозволом іншому учаснику брати задачу.

| Задача | Стан | Що має бути підтверджено |
|---|---|---|
| CH-14 | pending, робота Назара | Final training/data-strategy validation, test predictions, validator receipts, uploads, score вище baseline і вибір двох submissions |
| CH-16 | BLOCKED через CH-14; G10 ще відкритий | Clean-environment reproduction, повний rubric audit, власні висновки студентів, final freeze/tag і Pages verification |

Оновлювати після CH-14: розділи **10, 12–14**, результати final checkpoint і публічні scores, а не вже прийняте порівняння rubric моделей. Перед захистом обидва студенти мають пояснити висновки своїми словами. Незалежні CPU/static перевірки документації не означають завершення CH-16 або виконання G10.
