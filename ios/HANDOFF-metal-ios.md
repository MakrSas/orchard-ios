> Устарело: актуальное состояние порта — в `ios/PORTING.md`.

# Передача: Metal-бэкенд под iOS — готов

Параллельная сессия закончила порт `reims-vgpu`'s `backend-metal` на
`aarch64-apple-ios`. Тебе это НЕ нужно делать — только забрать.

## Забрать

Патч уже проверен `git apply --check` против твоего текущего дерева
(`reims-vgpu` с applied 0003 + untracked `vendor/qemu`) — ложится чисто:

    cd reims-vgpu
    git apply ../.claude/worktrees/metal-ios-port-6c8c87/patches/reims-vgpu/0004-metal-backend-apple-not-macos.patch

Патч самодокументирован: Subject + разбор каждого решения + список
непроверенного. Прочитай его шапку, там причины, а не список строк.

## Что сделано

`backend-metal` больше не привязан к `target_os = "macos"` — теперь
`target_vendor = "apple"`. 17 файлов, 124 гейта вида
`all(feature = "backend-metal", target_os = "macos")` сведены к
`feature = "backend-metal"`, платформенные гейты с модулей
`backend/metal/` сняты, Apple-зависимости в `Cargo.toml` перевешены на
вендора, `heap_placement.rs` — тоже на вендора.

Проверено:
  - `cargo build --release --target aarch64-apple-ios --no-default-features
    --features backend-metal -p reims-vgpu` — чисто; объектники несут
    `LC_VERSION_MIN_IPHONEOS` (device, не симулятор).
  - clippy на том же таргете новых предупреждений не даёт.
  - macOS не сломан: на `aarch64-apple-darwin` замена тождественна
    (`target_os = "macos"` там истинно по определению). 1391 тест проходит.
    Два падения (`spirv_bind`, 15 mesh-in-ICB) воспроизведены на чистом
    baseline — они были до этой работы.

Настоящих различий Metal API править не пришлось ни одного: бэкенд уже
написан под unified-memory подмножество Apple Silicon (только `Shared`,
ни одного managed-селектора, ни одного feature-set запроса).

`window.rs` / `host-window` и Vulkan-рельс не тронуты.

## Измерено на живом железе (Apple A16)

`tools/ios-metallib-probe` — приложение-пробник, собирается в IPA.

  1. **Гостевой macOS-metallib грузится на iOS-драйвере как есть.** Это был
     главный риск порта — закрыт положительно. Перештамповка платформенного
     байта не нужна.
  2. **BC-форматов на iPhone нет.** A16 даёт `supportsBCTextureCompression
     == false` при `apple6 apple7 apple8`; M1 даёт `true` при `apple6
     apple7`. BC привязаны к Mac, а не к номеру семейства — ни на одном
     iPhone их нет.

## Что НЕ сделано и почему

Гостевая macOS считает себя Маком и BC-текстуры шлёт. Vulkan-рельс это
переживает (`block_compressed` в `runtime/draw/texture_view.rs` — измеряемая
capability + типизированный отказ). **У Metal-рельса гейта нет** —
`backend/metal/format.rs` про BC не знает, потому что на Маках ответ всегда
был «да».

Metal на неподдерживаемый формат отвечает `abort()` процесса, а не nil. То
есть первая BC-текстура из гостя = вылет, а не отказ.

Гейт сознательно не добавлен: это изменение поведения device model, а не
механика порта, и ему место в отдельном патче. Если возьмёшься — образец
рядом: `constants.rs` ровно так же проверяет индексы ДО вызова, потому что
Metal отвечает на выход за диапазон исключением, которое роняет процесс.

## Не путать

Твой текущий блокер (KVM-заглушки `kvm_arm_pmu_init` не линкуются в
shared_lib, `target_stubs` в QEMU 11.1) — C/QEMU-сторона, к этой работе
отношения не имеет. Rust-сторона собирается независимо и уже зелёная.

VM у пользователя ещё нет — ждёт переноса данных на флешку и освобождения
места. Так что живого guest-прогона по iOS-пути пока не было ни у кого, и
пункты про `heapTextureSizeAndAlign` и device-capability аборты в патче
остаются непроверенными честно.
