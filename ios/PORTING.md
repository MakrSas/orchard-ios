# Orchard: порт на iOS

_Живой журнал этой ветки работы. Обновляется по ходу, не итоговый отчёт._

## Обзор

Orchard — форк QEMU, грузящий немодифицированный arm64 macOS (Ventura) через
настоящую цепочку загрузки Apple (`AVPBooter → iBootStage1 → iBootStage2 →
XNU`), эмулируя устройство `apple-vm` (vmapple) — то же, что
`Virtualization.framework` использует для VM на Apple Silicon. Сейчас
работает только на Linux/x86-64 через TCG, рабочий стол рендерится через
`reims-vgpu` (Vulkan, паравиртуальный GPU Apple).

Цель этой ветки — портировать сам эмулятор на iOS, чтобы гонять ту же macOS
VM прямо на iPhone, в бюджете ~2 ГБ ОЗУ.

**Отличие от Inferno.** Inferno (ChefKissInc + локальный форк пользователя) —
независимый проект, эмулирующий настоящий iPhone (SEP, kernelcache,
DeviceTree, персонализация через рестор-цепочку Apple), уже портированный
автором на iOS с рабочим JIT. Из Inferno-iOS берутся только универсальные
куски интерфейса и JIT-обвязка (`JIT.swift`, `QemuBridge.swift`) — под
капотом будет Orchard/QEMU с vmapple, а не эмуляция iPhone.

## Статус на сейчас

- **[ios/jit-probe](jit-probe)** — минимальное отдельное приложение, только
  `JIT.swift` + `LogCapture.swift` + `L10n.swift`. Подтвердило: JIT работает
  через mirrored mapping (split-WX) под StikDebug; `MAP_JIT` сам по себе
  недоступен (ожидаемо для сайдлоада). Риск: реальное исполнение
  сгенерированного кода на этом пути виснет намертво под трассировщиком —
  синтетическую кнопку-тест убрали, реальное доказательство работы JIT — сам
  QEMU (см. следующий пункт).
- **[ios/app](app)** — полная копия Inferno-iOS. Переименована в Orchard
  (`com.makr.orchard`, отдельно от настоящего Inferno). Из интерфейса
  убраны: экран рестора iPhone (`SetupView` → заглушка «VM не настроена»),
  пункты меню «Патчи» (менеджер пакетов/каталог/`.deb`/respring), «Кнопки
  устройства», настройки «Восстановление», «Батарея гостя» и «Строка
  состояния гостя» (обе — эмуляция iPhone-специфичного железа для гостя,
  не имеют смысла для macOS). В «Благодарностях» поправлена атрибуция
  (было «Inferno — форк QEMU», должно быть «Orchard»). Собирается и
  ставится, реально запускает JIT-диагностику.
  **Ещё видно iPhone-специфику, но не убрано** (в `Settings.swift`):
  выбор экрана как у iPhone 11/8/SE, текст про «одно ядро под SEP» и
  потолок процесса 3 ГиБ — трогать вместе с переписыванием `VMConfig`
  (пункт 5 ниже), не по отдельности.
- Готовится macOS Ventura VM через UTM (backend **Virtualize**, не QEMU) —
  нужна как источник персонализированного диска+NVRAM+ECID вместо
  скачивания готового tart-образа с ghcr.io.

## Архитектура (из каких частей состоит порт)

| Часть | Что | Статус |
|---|---|---|
| `ios/app` | Копия Inferno-iOS, SPM-стиль без Xcode-проекта | Чистится от iPhone-специфики |
| `ios/jit-probe` | Минимальный JIT-тест | Работает |
| `orchard/qemu` | Форк QEMU с vmapple/PAC-патчами | Только Linux x86-64 |
| `reims-vgpu` | Rust/Vulkan рендер рабочего стола | Нужна замена под iOS |

`QemuBridge.swift` (dlopen дилиба + `qemu_init`/`qemu_main_loop` на своём
треде) и `JIT.swift` — машино-независимые, переиспользуются почти как есть.
`VMConfig.swift` строит командную строку под iPhone/SEP — придётся
переписывать под vmapple-аргументы.

## Открытые блокеры и следующие шаги

1. Доставить macOS VM в UTM → написать конвертер `.utm`-бандла в
   `images/disk.raw` + `images/aux.img` + `config.json`, как ждут скрипты
   Orchard.
2. Прогнать `extract-avpbooter.py` / `patch-avpbooter.py` на полученном
   диске (на маке даже проще — APFS уже смонтирован нативно).
3. **Кросс-компиляция `orchard/qemu` под iOS-arm64.** Тулчейн есть готовый
   (`~/inferno-ios/cross-ios-arm64.txt`, тот же, что использовал Inferno),
   но сама компиляция именно vmapple-патчей — не пройденная территория.
4. **Замена `reims-vgpu`** на программный framebuffer — Vulkan-путь на iOS
   не встанет. `VMModel` в `InfernoApp.swift` уже имеет абстракцию
   `GuestDisplay` с `EmbeddedDisplay` (кадры напрямую из библиотеки, без
   сети) — правильная точка подключения, но саму связку внутри QEMU ещё
   разбирать по `TECHNICAL.md`.
5. Переписать `VMConfig.arguments()` под `-M apple-vm,uuid=...` вместо
   iPhone/SEP аргументов.
6. Полная зачистка мёртвого iPhone-кода (`Restore*.swift`, `IPSW.swift`,
   `SEPFirmware.swift`, `Cryptex1.swift`, `IMG4.swift`, `Ticket.swift`,
   `GuestPackages.swift`, `AppCatalog.swift`, `CatalogView.swift`,
   `PackagesView.swift`, `Repos.swift`, `GuestInstaller.swift`) — сейчас
   намеренно оставлены нетронутыми (компилируются, но не вызываются из UI),
   потому что `VMConfig.arguments()` всё ещё на них ссылается; удалять
   безопасно только вместе с пунктом 5.

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
  отдельный `build-ios/` через `meson setup --cross-file=scripts/cross-ios-arm64.txt -Dshared_lib=true` с теми же опциями, что использовал
  Inferno (`-Dkvm=disabled -Dhvf=disabled -Dgtk=disabled -Dsdl=disabled
  -Dvnc=enabled -Dcoroutine_backend=ucontext` и т.д.), подсунув
  предсгенерированные `.mak`-файлы. Сейчас крутится bootstrap-`configure`.
- Cross-file лежит в [scripts/cross-ios-arm64.txt](../scripts/cross-ios-arm64.txt)
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
в `[properties]` [scripts/cross-ios-arm64.txt](../scripts/cross-ios-arm64.txt)
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
