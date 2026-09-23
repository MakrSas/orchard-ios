# Orchard: порт на iOS

_Живой журнал этой ветки работы. Обновляется по ходу, не итоговый отчёт._

## Обзор

Orchard — форк QEMU, грузящий немодифицированный arm64 macOS (Ventura) через
настоящую цепочку загрузки Apple (`AVPBooter → iBootStage1 → iBootStage2 →
XNU`), эмулируя устройство `apple-vm` (vmapple) — то же, что
`Virtualization.framework` использует для VM на Apple Silicon. Апстрим
Orchard работает на Linux/x86-64 через TCG, рабочий стол рендерится через
`reims-vgpu` (Vulkan, паравиртуальный GPU Apple). На iOS рендер идёт через
Metal-бэкенд того же `reims-vgpu`.

Цель этой ветки — портировать сам эмулятор на iOS, чтобы гонять ту же macOS
VM прямо на iPhone, в бюджете ~2 ГБ ОЗУ.

**Отличие от Inferno.** Inferno (ChefKissInc + локальный форк пользователя) —
независимый проект, эмулирующий настоящий iPhone (SEP, kernelcache,
DeviceTree, персонализация через рестор-цепочку Apple), уже портированный
автором на iOS с рабочим JIT. Из Inferno-iOS берутся только универсальные
куски интерфейса и JIT-обвязка (`JIT.swift`, `QemuBridge.swift`,
`ui/inferno-embed.c`, iOS-правки QEMU для JIT и корутин) — под капотом
Orchard/QEMU с vmapple, а не эмуляция iPhone. Исходники Inferno лежат
локально в `~/inferno-ios/src/inferno` (форк `MakrSas/Inferno`), приложение —
`MakrSas/Inferno-iOS`.

## Где мы сейчас — 2026-09-23, 04:30

**macOS Ventura 13.6 на iPhone 15 входит в систему и работает на рабочем столе,
около 15 кадров/с.** Настройки: память 2 ГБ, буфер трансляций 128 МБ, 2 ядра,
960×540, 30 Гц. Системные настройки гостя пока открываются с трудом; сеть не
проверена.

Что дало результат, по убыванию веса:

- **Клавиатура: `conditional-intr-mapping` на `apple-vm`** (`hw/vmapple/vmapple.c`).
  `machine_class_base_init()` даёт каждому классу машины пустой `compat_props`,
  поэтому `apple-vm` не наследовал настройку `vmapple`. Без неё macOS кладёт
  события HID в кольца 1/2, а INTx поднимается только для кольца 0; гость
  находил их лишь по прерыванию MFINDEX wrap — раз в 2.048 с (issue #2705).
  Каждая клавиша «держалась» 2 с и повторялась автоповтором.
- **Утечка Metal**: autorelease pool на каждый вход в reims-vgpu (патч 0009).
  IOAccelerator рос до 280+ МБ и iOS убивала приложение по памяти.
- **Отрисовка reims-vgpu в своём потоке** (`hw/display/reims-vgpu-mmio.c`), как
  в PCI-варианте, а не в BH под BQL: цепочки по 84–104 мс держали BQL и ядра
  гостя.
- **Кеш сэмплируемых текстур** по содержимому (патч 0010): обои 6016×6016
  (138 МиБ) распаковывались и грузились в Metal на каждой отрисовке.
- **Вывод кадров в приложении** через `CALayer.contents` вместо SwiftUI `Image`
  (кадры копились в памяти графики), своя экранная клавиатура и строка ввода,
  клавиша целиком за один захват BQL (`orchard_input_key_tap`).
- **PAC**: кеш `aa64_va_parameters` и лёгкий impdef-хеш вместо xxhash
  (`target/arm/tcg/pauth_helper.c`): `pauth_computepac` 11 % → 2 %.
- **Память**: меньше 2 ГБ гость зацикливается на перезагрузке; у установки нет
  `increased-memory-limit`, потолок 3 ГБ; кеши reims-vgpu урезаны для iOS
  (патч 0008). В `emulator.log` строки «Память» с разбивкой по регионам и
  профиль CPU по функциям.
- **Экран**: выбор разрешения (только выбранное и меньшие, иначе macOS
  возвращает сохранённый 1920×1080) и частоты 120/60/30 Гц (патч 0007);
  исправлено двоение картинки при уменьшении разрешения (`orchard_display_read`).

Грабли: `overlay.qcow2`, переживший десятки прерванных загрузок, перестал
принимать пароль на экране входа (поля Name/Password на английском) — чистая
конвертация (`scripts/utm-to-orchard.py`) это сняла. Сначала менять overlay,
потом искать в коде. `reims-vgpu-fail.log` теперь начинается заново при каждом
запуске (прошлый — `.prev.log`).

## Было — 2026-09-22, 23:00

**macOS Ventura 13.6 загружается на iPhone 15 до экрана входа, с картинкой.**
Видно яблоко с полоской загрузки, затем loginwindow (поля «Name» / «Enter
Password», Shut Down / Restart / Sleep). Около 1 кадра/с на экране входа.

Устройство: iPhone15,4 (A16, 6 ГБ), iOS 27.0 beta (24A5390f), JIT через StikDebug,
память гостя 2 ГБ.

Как дошли до картинки — последний барьер был в шейдерах:

- iOS-драйвер Metal отвергал каждый MTLB, скомпилированный гостем:
  `This library format is not supported on this platform (or was built with
  an old version of the tools)`. Все отрисовки падали и заменялись очисткой —
  WindowServer рисовал, но кадры были чёрные.
- Патч `patches/reims-vgpu/0006` на iOS повторяет загрузку с переписанным
  платформенным штампом (байт `0x0B` → `0x82`, сброс бита 7 в `0x05`) — и
  **все 14 блобов загрузились** (`metal_library_restamp … result=loaded`), после
  этого в прогоне нет ни одного `draw_encode_fail` / `draw_fail_clear_fallback`.
- Любопытно: у гостевых блобов парсер видит `platform=unknown os=0.0`, то есть
  компилятор Ventura пишет заголовок другой версии, чем у библиотек, по которым
  выводилась раскладка. Правка тех же байт всё равно сработала — если
  понадобится разбираться, блобы лежат в `Documents/reims-vgpu-mtlb/`.

### Открытые проблемы — с этого начинать (состояние на 2026-09-22, 23:00)

Последняя сборка — **22:58** (коммит `0015f3f`). Отчёт пользователя по ней ещё не получен.

**1. Рабочий стол рисуется уменьшенным в левом нижнем углу кадра.**
Обои и loginwindow занимают только часть кадра 1920×1080 — по двум скриншотам
с телефона **≈0.82 ширины × 0.50 высоты, прижато к левому нижнему углу**
(измерено: 697×238 px картинки внутри рамки 848×477). Остальное — чёрное, но
курсор (USB-планшет, абсолютные координаты) ходит по всему кадру, то есть гость
считает экран полным 1920×1080. Цвета правильные.
Что известно из `reims-vgpu-fail.log`:
  - все present'ы идут с `geom=1920x1080`;
  - `surface_row_native_format format=0x73 tight=15360 rgba_stride=7680
    cache_is_native=false publication=RailResident mapping=1` — рабочий стол
    гостя живёт в `RGBA16Float` (0x73, 8 байт/пиксель), и путь публикации
    конвертирует его в RGBA8 для консоли;
  - `display_vbl … not_enabled=… arm=not_enabled` и
    `display_present_signal … not_enabled=1` растут всё время — дисплей со
    стороны гостя, похоже, так и не переходит в «enabled»;
  - `released_pages reason=released_write_after_release` — устройство пишет в
    страницу после того, как гость её освободил (отдельная проблема корректности).
Гипотезы, по убыванию вероятности:
  (а) геометрия отрисовки на Metal-пути под iOS: viewport/scissor или размер
      резидентного таргета не совпадает с тем, что считает гость (ровно 0.5 по
      вертикали наводит на двукратный масштаб — HiDPI/«Retina»-режим или
      перевёрнутая ось Y + половинный viewport);
  (б) конвертация `RGBA16Float → RGBA8` в публикации/скан-ауте
      (`backend/metal/resident.rs` `read_published_rgba8`,
      `runtime/scanout/metal.rs`, `reims_vgpu_qemu_scanout_copy`);
  (в) незавершённое согласование режима дисплея (см. `not_enabled` выше) —
      WindowServer мог выбрать режим меньше, чем 1920×1080.
Как проверить: сравнить с эталонным путём «macOS-хост + Metal» из таблицы
Supported pathways в `reims-vgpu/AGENTS.md`; сохранить один опубликованный кадр
(RGBA16F-исходник и RGBA8-результат) в `TMPDIR` и посмотреть глазами; вывести в
журнал viewport'ы первых отрисовок кадра и размер резидентного таргета.

**2. Вылет через 5–7 с после появления экрана входа.**
В `reims-vgpu-fail.log` перед вылетом ошибок нет, в `emulator.log` — просто
обрыв, без «Shutdown». Прогон был с памятью гостя **2 ГБ** и буфером
трансляций 128 МБ — похоже на jetsam (iOS отбирает память; потолок процесса
около 3 ГБ). Сборка 22:58 даёт выбрать **1.5 ГБ** и ставит буфер 256 МБ по
умолчанию. Подтверждение — файл `JetsamEvent-*.ips` из «Данных аналитики».
Замер пользователя: с **2 ГБ** курсор появляется на 1:10, с **1.5 ГБ** — не
появился и за 1:50 (гостю тесно, он, видимо, упирается в сжатие памяти).
Поэтому добавлено **1.75 ГБ** — середина между «не грузится» и «вылетает».
**Разобрано по логам 2026-09-23:** при 1.5 и 1.75 ГБ гость не «тесный», а в
цикле перезагрузок — iBoot2 передаёт управление ядру, через ~6 с машина
молча перезапускается (17 проходов iBoot за 2.5 мин, `device_reset seq=2…17`
в `reims-vgpu-fail.log` каждые ~9 с, паники в консоли нет). Причина не
найдена; macOS-гостю теперь предлагаются только 2 / 2.5 / 3 ГБ. На 2 ГБ при
960×540 отрисовка чистая (23 шейдера загружены перештамповкой, ни одного
отказа), процесс обрывается на ~92 с без ошибок — похоже на jetsam. В
`emulator.log` теперь строки «Память: …, до потолка …» каждые +100 МБ.

**3. Очень низкий FPS** — около 1 кадра/с на экране входа. Эмуляция всего Мака
под TCG без гипервизора. В сборке 22:58 буфер трансляций 256 МБ (по замерам
Inferno на телефоне 256 МБ дают в 2–3 раза больше кадров, чем 64). Дальше —
профилировать: сэмплер в `emulator.log` показывает много `pauth_*`,
`helper_lookup_tb_ptr`, `tcg_gen_code`, `get_phys_addr*`.
Рычаги без увеличения буфера (сборка 2026-09-22, ещё не замерены):
  - Настройки → Экран: **разрешение** (1920×1080 / 1600×900 / 1280×720 /
    960×540, `REIMS_VGPU_DISPLAY_MODE`) и **частота** (120 / 60 / 30 Гц,
    `REIMS_VGPU_DISPLAY_HZ`). Раньше гость всегда рисовал 1920×1080 на 120 Гц:
    WindowServer подстраивает отрисовку под частоту экрана, и reims-vgpu слал
    120 прерываний VBL в секунду;
  - Настройки → Машина: **2 и 3 ядра** для macOS-гостя (минимум 4 был из-за SEP
    гостя-iPhone). У телефона 2 производительных ядра — сравнить 2/3/4;
  - `scripts/guest-tune.sh` — запустить один раз в госте в UTM на Маке
    (Spotlight, анимации, прозрачность, автообновления), затем включить
    автовход и заново сконвертировать VM целиком (старый `overlay.qcow2` не
    оставлять).
  - Сборка 00:09: картинка при 960×540 двоилась и сжималась в верхнюю
    четверть — `orchard_display_read` сообщал о смене размера, только когда
    буфер приложения *меньше* кадра, а при уменьшении экрана он больше.
    Теперь — при любом несовпадении. Та же ошибка возможна в проблеме 1
    («картинка в верхней половине»), если загрузочный кадр был больше рабочего
    стола; проверить на новой сборке.

**4. Клавиатура** (сборка 22:43, коммит `6ebf7c7`) — ещё не проверена: кнопка ⌨︎
слева внизу; экранная печатает через US-раскладку, аппаратная передаёт HID-коды.

## Как собрать и запустить

1. **Библиотека:** `scripts/build-ios.sh` → `qemu/build-ios/libqemu-aarch64-softmmu.dylib`.
   Нужны зависимости в `deps/ios` (glib, pixman, libslirp, libucontext…:
   `scripts/fetch-ios-deps.sh` качает их архивом из релиза `ios-deps-1`, или
   `ORCHARD_IOS_DEPS=` на свой префикс), Xcode с iPhoneOS SDK (cross-файл
   собирается из `scripts/cross-ios-arm64.txt.in` под установленный Xcode),
   `rustup target add aarch64-apple-ios`. Первый раз ~20–30 мин на M1 8 ГБ,
   дальше инкрементально.
2. **Приложение:** `ios/app/build.sh` → `ios/Orchard.ipa` (библиотеку ищет
   сначала в `qemu/build-ios`).
3. **VM:** выключить её в UTM, затем
   `scripts/utm-to-orchard.py ~/Library/Containers/com.utmapp.UTM/Data/Documents/macOS.utm --out-dir ~/OrchardVM`.
   Получается `disk.qcow2` (~15 ГБ), `overlay.qcow2`, `aux.img`,
   `AVPBooter.patched.bin`, `config.json`. Писать на внутренний APFS, на
   флешку копировать потом Finder'ом (см. грабли про панику ядра).
4. Папку скопировать на внешний диск («Образы», exFAT) или в «Файлы» →
   Orchard → OrchardVM. В приложении: Настройки → «Где лежит VM» → «Выбрать
   папку…» — диск используется на месте, через security-scoped закладку.
5. Запускать через StikDebug. Память 1.5–2 ГБ (пользователь ставит 2G;
   потолок процесса ~3 ГБ), 4 ядра, tb-size 128.
6. **Логи** — «Файлы» → «На iPhone» → Orchard:
   - `emulator.log` / `emulator.prev.log` — приложение и stderr QEMU; каждые
     30 с `CPU0 PC=… EL… · ядер в работе`, каждые 15 с `Экран: …`;
   - `guest-console.log` — serial гостя;
   - `reims-vgpu-fail.log` — канал отказов reims-vgpu (главный источник по картинке);
   - `reims-vgpu-mtlb/` — отвергнутые драйвером шейдеры;
   - крэши — Настройки → Конфиденциальность → Аналитика → `Orchard-*.ips`.

## Что уже починено (коммиты в `main`, по порядку)

| Коммит | Что |
|---|---|
| `39e5fc5` | чекпоинт всего порта (до этого жил незакоммиченным) |
| `9aba578` | запуск macOS-гостя из приложения: `ui/inferno-embed.c`, `VMConfig` под `-M apple-vm`, выбор папки VM, `build-ios.sh`, `utm-to-orchard.py`; в `reims-vgpu-mmio.c` — `gfx_update → bool`, `vm_*` вместо запрещённого на iOS `mach_vm.h`, без перехвата `qemu_main` в библиотеке |
| `0b0caf4` | JIT на iOS 26+: буфер «благословляет» отладчик (`brk #0x69`, из Inferno) |
| `4ab3138` | корутины через `libucontext` (системные `getcontext` на iOS — заглушки → `abort`) |
| `0155ae2` | `virtio-net-pci,romfile=` (нет `efi-virtio.rom` в бандле) |
| `ba95e99` | снимок CPU через QMP каждые 30 с |
| `afe3eef` | строка про экран: кадры показаны / дошли / чёрный ли |
| `5a968be` | журнал reims-vgpu на iOS в `TMPDIR` (патч 0005), приложение копирует в Documents; исправлен снимок CPU |
| `1c901d1` | текст ошибки Metal, сохранение блобов, повтор с iOS-штампом (патч 0006) — **дал картинку** |
| `6ebf7c7` | клавиатура: `inferno_input_key_hid`, `GuestKeyboard.swift`; без скругления экрана для macOS |
| `0015f3f` | выбор памяти 1.5 ГБ (4 ГБ убран), буфер трансляций 256 МБ по умолчанию |

**reims-vgpu** лежит прямо в дереве (с 2026-09-23, как и в upstream Orchard): сабмодуля
и `patches/` больше нет. Правки, которые раньше были патчами 0003…0012 (Metal на iOS,
режимы экрана, кеши, autorelease-пулы, кеш сэмплируемых текстур, размер PSO), вшиты в код;
ниже по тексту номера патчей остались как история.

## Факты и грабли, которые стоили времени

- **UTM на macOS 26+ хранит диск в ASIF** (магия `shdw`), QEMU его не читает.
  Конвертер подключает образ только на чтение (`diskutil image attach`) и
  пишет qcow2.
- **«pflash»-диски vmapple — не флеш**, а блочные бэкенды для загрузочного
  PV-устройства (BDIF), поэтому qcow2 годится и там.
- **AVPBooter брать из гостя**, а не с Мака: у macOS 27 место патча уехало
  (десять кандидатов с тем же прологом), у Ventura — ровно `0x2314`.
- `AuxiliaryStorage` UTM = формат tart (заголовок `0x4000`), ECID — в
  `MachineIdentifier` из `config.plist` (bplist). Старый `images/config.json`
  от tart к этой VM не подходит.
- **JIT на iOS 26+** под Trusted Execution Monitor: RW→RX изнутри процесса
  убивает его молча. Только последовательность UTM/Inferno с отладчиком.
- **На Darwin нет OFD-локов**, поэтому `locking=auto` в QEMU = без блокировок;
  оверлей нужен, чтобы диск открывался дважды только на чтение и оставался
  нетронутым.
- **`/tmp` в песочнице iOS не пишется** — всё, что reims-vgpu туда писал, пропадало.
- **Заголовок MTLB** (по ~900 образцам): `[0x0B]` платформа — `0x81` macOS,
  `0x82` iOS, `0x87` симулятор; бит 7 в `[0x05]` повторяет то же; `[0x08]` —
  версия AIR; `[0x0C]`/`[0x0E]` — major/minor целевой ОС. Вывод из образцов, не документация.
- **На iPhone нет BC-сжатия текстур** (A16: `supportsBCTextureCompression = false`,
  на M1 — true): BC привязано к Mac, а не к семейству GPU. В Metal-рельсе
  reims-vgpu гейта на BC нет — как только гость пришлёт BC-текстуру, Metal
  уронит процесс `abort()`-ом.
- **Пустой экран на стадии iBoot ожидаем**: на MMIO-варианте reims-vgpu никто
  не регистрирует ранний framebuffer (`reims_vgpu_mmio_set_early_fb` не вызывается).
- `query-cpus-fast` показывает один `thread-id` на все vCPU — на Darwin
  `qemu_get_thread_id()` возвращает pid, это не однопоточный TCG.
- `ядер в работе: 1 из 1` в снимке CPU — вероятно, особенность вывода HMP
  `info cpus`, счётчику пока не верить.
- **Конвертация ASIF с чтением `/dev/rdisk` и одновременной записью на exFAT
  уронила macOS 27 beta** (паника `SPTM VIOLATION_ILLEGAL_UNMAP`). Писать на
  внутренний APFS, потом копировать — без проблем.
- Флешка `OrchardVM` («OnlyDisk», USB) пишет меньше 1 МБ/с — не использовать.
  Рабочий диск — «Образы» (exFAT, ~26 МБ/с запись).
- Хук этой среды не даёт сессии в worktree писать в основной чекаут; работа
  идёт в worktree на ветке `claude/metal-ios-port-6c8c87`, после каждого
  коммита `main` в основном чекауте перематывается fast-forward'ом.

## Открытые вопросы, кроме картинки

1. BC-гейт в Metal-рельсе reims-vgpu (см. выше) — иначе будущий `abort()`.
2. Клавиатура: только US-раскладка для экранной (аппаратная — любая, раскладку
   выбирает гость). Русский ввод с экранной клавиатуры не поддержан.
3. Текстовый лог ядра: только через снижение безопасности в самой VM (UTM,
   recoveryOS) и повторную конвертацию.
4. Айфонные остатки в приложении (выбор «экрана iPhone 11», `Restore*.swift` и т.д.).
5. Производительность — пока не мерили.
6. Лицензии: `ui/orchard-embed.c` пришёл из Inferno-iOS (там `ui/inferno-embed.c`, AGPL-3.0), здесь он, как и приложение, GPL-2.0-or-later; остальной
   QEMU — GPL-2.0-or-later. Изменения в `qemu/` по правилам QEMU (`qemu/AGENTS.md`)
   годятся для этого форка, но не для отправки в апстрим qemu-devel.

## Архитектура

| Часть | Что | Статус |
|---|---|---|
| `ios/app` | приложение (оболочка из Inferno-iOS), SPM-стиль без Xcode-проекта | запускает macOS-гостя |
| `ios/jit-probe` | минимальный JIT-тест | работает |
| `qemu/` | форк QEMU 11.1 с vmapple/PAC-патчами, библиотекой `shared_lib` | собирается под iOS, грузит macOS |
| `reims-vgpu` | паравиртуальный GPU Apple, бэкенд Metal | работает на iOS (шейдеры гостя через перештамповку) |
| `tools/ios-metallib-probe` | пробник загрузки metallib на устройстве | работает |

Путь картинки: reims-vgpu (Metal) → `DisplaySurface` консоли QEMU →
`ui/orchard-embed.c` → `EmbeddedDisplay` в приложении. Ввод: касание →
`orchard_input_touch` → USB-планшет vmapple.

---

# История: как шли к первой сборке

### Кросс-компиляция orchard/qemu под iOS — в процессе

Начали пункт 3 напрямую. Находки:

- **`qemu_init`/`qemu_main_loop`/`qemu_cleanup` — это апстримные функции QEMU**
  (`system/vl.c`, `system/runstate.c`), не инферно-специфичные. Значит
  готовый С-API для встраивания в приложение (то, что дёргает
  `QemuBridge.swift`) есть и у orchard/qemu из коробки.
- **Сборка как dylib — не апстримная фича.** У Inferno в `meson.build` и
  `meson_options.txt` есть свой option `shared_lib` (переключает
  `static_library`/собранную в экзешник → `shared_library` +
  тонкий `qemu-system-<target>`, который её дёргает). Портировали этот же
  паттерн в **orchard/qemu** (`meson_options.txt`, секция сборки таргета в
  `meson.build`) — для обычной линукс-сборки ничего не меняется
  (`shared_lib` по умолчанию `false`).
- **GPU на iOS — не headless-заглушка, а Metal.** `reims-vgpu` уже имеет
  готовый бэкенд `backend-metal` (родной для Apple-платформ, полная
  реализация — `src/backend/metal/`, `runtime/*/metal.rs`), а
  `hw/vmapple/vmapple.c` сам умеет выбрать GFX-реализацию через
  `object_class_by_name` в рантайме. Поправили `hw/display/meson.build`:
  при `host_machine.subsystem() == 'ios'` cargo собирает
  `--features backend-metal` с `--target aarch64-apple-ios` вместо
  `--features backend-vulkan,host-window`. `rustup target add
  aarch64-apple-ios` — сделано.
- **`configure` для orchard/qemu не выпилен** (в отличие от Inferno — там
  `git log` прямо показывает коммит «remove configure script», а
  Kconfig-таргет-файлы (`config-host.mak`, `TARGET_DIRS` и т.д.) заранее
  сгенерированы и просто читаются `meson.build` — `configure` дальше не
  нужен). У нас `configure` есть, и он реально нужен как минимум один раз:
  `meson.build` не генерирует `config-host.mak` сам, только читает
  (`keyval.load(...)`), Kconfig-резолюцию через `scripts/minikconf.py`
  запускает `meson setup`, но список таргетов (`TARGET_DIRS`) в
  `config-host.mak` только `configure` пишет.
- **План:** прогнать `configure` нативно (macOS, без всякого iOS) в
  отдельной `build-bootstrap/` только чтобы получить
  `config-host.mak`/`aarch64-softmmu-config-target.mak`, затем поднять
  отдельный `build-ios/` через `meson setup --cross-file=<build-ios>/cross-ios-arm64.txt (из scripts/cross-ios-arm64.txt.in) -Dshared_lib=true` с теми же опциями, что использовал
  Inferno (`-Dkvm=disabled -Dhvf=disabled -Dgtk=disabled -Dsdl=disabled
  -Dvnc=enabled -Dcoroutine_backend=ucontext` и т.д.), подсунув
  предсгенерированные `.mak`-файлы. Сейчас крутится bootstrap-`configure`.
- Cross-file лежит в [scripts/cross-ios-arm64.txt.in](../scripts/cross-ios-arm64.txt.in)
  — переиспользует уже собранный Inferno-тулчейн/префикс
  (`~/inferno-ios/prefix`: glib, gmp, pixman, lzfse, nettle, libpng,
  libslirp, libtasn1, libucontext — общие C-зависимости QEMU, не
  завязанные на эмуляцию iPhone).
- **Первый реальный прогон `meson setup` под iOS дошёл до содержательной
  ошибки** (не path/конфиг — сама проверка QEMU): `meson.build` жёстко
  исключает coroutine-backend `ucontext` для любого Darwin (`host_os !=
  'darwin'`), не различая настоящий macOS и iOS. Поправили на
  `host_machine.subsystem() == 'ios'` как исключение из исключения.
  Дальше — ещё одна настоящая (не портовая) проблема: SDK iOS требует
  `_XOPEN_SOURCE` для deprecated-ucontext-API (`#error The deprecated
  ucontext routines require _XOPEN_SOURCE to be defined`) — добавили
  `-D_XOPEN_SOURCE` в cross-file, руками проверили что пробник компилируется
  и линкуется.

**`meson setup` под iOS прошёл целиком, без единой ошибки.** Подтверждено
по выводу: `coroutine backend: ucontext`, `TCG backend: native (aarch64)`
(гость arm64 на хосте arm64 — как раз случай iPhone, без кросс-архитектурной
трансляции самого TCG), `target list: aarch64-softmmu`, `shared_lib: true`.
Граф сборки содержит `hw_vmapple_vmapple.c.o` и
`hw_display_reims-vgpu-mmio.c.o` — vmapple и reims-vgpu-glue подхватились.
Итоговая библиотека называется `libqemu-aarch64-softmmu.dylib` — ровно то
имя, что уже ищет `QemuBridge.swift`, менять на стороне Swift ничего не
нужно.

**Первая реальная ошибка компиляции (не наша, апстримный QEMU-код):**
`osdep.h` дёргает `pthread_jit_write_protect_np()` под голым `#ifdef
__APPLE__`, а её на iOS **нет вообще** — не просто скрыта в заголовке
(`__API_UNAVAILABLE(ios, ...)` в `pthread.h`), а реально отсутствует как
линкуемый символ (проверили руками — `ld: symbol(s) not found`). Apple
предлагает взамен `pthread_jit_write_with_callback_np`, но она тоже
привязана к `MAP_JIT`, который у сайдлоад-приложений недоступен без
энтайтлмента — то же самое ограничение, что уже видели в `JIT.swift`.
Решение позаимствовано у Inferno (у них в точности та же правка с тем же
обоснованием): на iOS `qemu_thread_jit_execute`/`qemu_thread_jit_write`
становятся no-op — там, где JIT вообще доступен (энтайтлмент или отладчик),
страницы `MAP_JIT` сразу RWX, переключать нечего. Сделали через
`TargetConditionals.h`/`TARGET_OS_IPHONE` вместо отдельного Kconfig-символа
(`include/qemu/osdep.h`).

**Прогресс — 727/1802 объектов собралось**, наткнулись на две новые
настоящие ошибки:

1. `block/file-posix.c`: `struct statfs`/`fstatfs` не объявлены. Скрытый
   баг апстримного QEMU — заголовки (`sys/param.h`, `sys/mount.h`)
   подключаются только в блоке под `HAVE_HOST_BLOCK_DEVICE` (реальный
   доступ к дискам через IOKit, на iOS осмысленно выключен), а более
   поздний блок `#if defined(__APPLE__) && (__MACH__)` использует их же,
   не проверяя это условие — на обычном macOS-билде `HAVE_HOST_BLOCK_DEVICE`
   почти всегда true, поэтому баг никогда не проявлялся. Добавили свой
   `#include` прямо в этот блок (`block/file-posix.c`).
2. **`reims-vgpu` не собирался с `--features backend-metal` в одиночку**
   (без `backend-vulkan`): пять `match`-веток в `backend/mod.rs` были
   помечены `#[cfg(all(feature = "backend-metal", target_os = "macos"))]`,
   тогда как сам вариант `SelectedBackend::Metal` объявлен только под
   `feature = "backend-metal"` (без `target_os`). На iOS (`target_os =
   "ios"`) вариант остаётся, а эти пять веток — нет, отсюда
   `non-exhaustive patterns`. Все пять методов у `MetalBackend` не
   переопределены — это no-op/`None` по умолчанию из трейта `Backend`, то
   есть платформенной специфики там нет и охранять было нечего. Убрали
   `target_os = "macos"`, привели к тому же виду, что и остальные ветки
   этого же варианта. Патч — раз `reims-vgpu` вендорится как сабмодуль
   («Not ours» в README), оформили по конвенции проекта:
   [patches/reims-vgpu/0003-metal-backend-not-macos-only.patch](../patches/reims-vgpu/0003-metal-backend-not-macos-only.patch).

Гоняли `ninja` в третий раз — реально дошли до 918/2132, но `reims-vgpu`
всё равно собирался с `backend-metal` и падал на той же
`compile_error!`, несмотря на `configs/devices/aarch64-softmmu/ios.mak`
(`CONFIG_REIMS_VGPU=n`).

**Решили пойти по-другому: настоящий Metal-порт `reims-vgpu` под iOS
делается отдельно** (не блокирует ядро порта) — см. `patches/reims-vgpu/`,
промпт для параллельной сессии сохранён отдельно. Здесь возвращаемся к
headless (без reims-vgpu вообще) — сначала проверить, что JIT/vmapple/PAC
вообще работают на устройстве через serial console, картинка — потом.

**Почему `ios.mak` не срабатывал.** `--with-devices-aarch64=ios` — флаг
`configure`, а не свойство cross-файла само по себе; `configure`
транслирует его в `[properties] aarch64-softmmu = 'ios'` своего
автогенерируемого `config-meson.cross`. Мы обходим `configure` и пишем
cross-file руками (как Inferno), так что это свойство никогда не
попадало в наш файл — `meson.build` (`config_input =
meson.get_external_property(target, 'default')`, строка ~3409) молча
брало `default.mak` вместо `ios.mak`. Добавили `aarch64-softmmu = 'ios'`
в `[properties]` [scripts/cross-ios-arm64.txt.in](../scripts/cross-ios-arm64.txt.in)
— заработало, `CONFIG_REIMS_VGPU` подтверждённо отсутствует в
сгенерированном `aarch64-softmmu-config-devices.mak`.

**Собралось — headless, но собралось.** Дошли до 2127/2128, упали на
последнем `.c`-файле: `hw/vmapple/vmapple.c` безусловно подключает
`reims-vgpu-shim.h` (сам он ничего оттуда не использует — GFX выбирается
рантайм-строкой через `object_class_by_name`), а тот тянет
`reims_vgpu_qemu_abi.h` из крейта, которого нет в этой сборке. Обернули
`#include` в `#ifdef CONFIG_REIMS_VGPU`.

Все `.c` собрались, упали на **линковке**: не хватало ARM KVM-стабов
(`kvm_arm_pmu_init` и т.д. — `target/arm/kvm-stub.c`). Причина —
версионный дрейф: паттерн `shared_lib` скопирован с дерева Inferno-iOS
(QEMU 10.2.2), а orchard/qemu — 11.1.50; там, откуда взят паттерн,
`target_stubs` попадал в библиотеку бесплатно через `arch_deps`, в этой
версии это отдельная зависимость, добавляемая только исполняемому файлу
(`qemu-system-aarch64`), не самой `shared_library()`. Добавили
`target_stubs` явно в зависимости `shared_library()` (`meson.build`).

## Результат: `libqemu-aarch64-softmmu.dylib` собрался

39 МБ, `Mach-O 64-bit ... arm64`, `LC_BUILD_VERSION platform 2 (iOS) minos
16.0`, экспортирует `qemu_init`/`qemu_main_loop`/`qemu_cleanup` — ровно то,
что `QemuBridge.swift` ждёт. Headless (без reims-vgpu/картинки) — это
первая, самая важная проверка: JIT + vmapple + PAC-патчи orchard/qemu
вообще компилируются и линкуются под iOS. Работает ли это в рантайме на
устройстве (запуск, serial-консоль) — следующий шаг, ещё не проверяли.

Параллельно на отдельной сессии идёт настоящий Metal-порт `reims-vgpu` под
iOS (промпт сохранён, патчи будут в `patches/reims-vgpu/000N-*`) — когда
будет готов, headless отключение снимем (`ios.mak` → `CONFIG_REIMS_VGPU=y`,
`aarch64-softmmu = 'ios'` в cross-file уже на месте).

## Решения и грабли (лог по ходу работы)

- **macOS ≠ iOS по срокам подписи.** Apple держит подпись на старые версии
  macOS куда дольше, чем на iOS (SHSH-окно). Ventura всё ещё подписывается —
  можно ставить конкретную нужную версию через UTM вручную с IPSW, а не
  только «последнюю доступную».
- **UTM: только backend «Virtualize».** Тот, что использует
  `Virtualization.framework` и даёт настоящую персонализацию (LocalPolicy,
  ECID). QEMU-backend в UTM создаёт обычный generic `aarch64/virt`, это не
  то.
- **Причина `Installation failed` в UTM оказалась не диском.** Сначала
  подозревали ExFAT (AppleDouble-файлы `._*`, нет sparse) — переехали на
  APFS, ошибка осталась той же. Реальная причина: Screen Time /
  `familycontrols` держит системный DNS-перехватчик (`utun` на
  `198.18.0.0/15`), который подменял ответ для `gs.apple.com` — серверу
  персонализации Apple. `profiles remove`/выключение тумблеров/`killall
  parentalcontrolsd`/перезагрузка не снимали перехват. Обошли через прямой
  DoH-запрос (`curl https://1.1.1.1/dns-query?...`, минуя обычный DNS) →
  реальный IP → `/etc/hosts`.
- **Требования Apple Virtualization framework.** VM с 1 ядром / 2 ГБ падает
  на валидации — нужно от 4 ядер / 4 ГБ (на 8-гигабайтном маке).
- **Finder не умеет копировать sparse-файлы между томами.** Требует места
  под полный «кажущийся» (логический) размер, а не реальный на диске — у
  дисковых образов VM это может быть в разы больше (36 ГБ реальных vs
  105 ГБ кажущихся в одном случае). `rsync -S` тоже не спасает: на APFS
  `fpathconf` не поддерживает то, на чём держится sparse-детект rsync.
  Работает `ditto`.
  **Поправка позже:** при межтомовом копировании `ditto` дыры тоже
  разворачивал — на флешке `root.bothpatches` занял полные 32 ГиБ вместо
  8.4, и копирование упёрлось бы в место. Для образов VM проще не полагаться
  на разреженность вовсе и хранить их в qcow2 (так и делает `utm-to-orchard.py`).
