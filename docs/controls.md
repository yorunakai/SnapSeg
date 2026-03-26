# Controls

## Mouse

- Left click: positive point
- Right click: negative point
- Mouse wheel: zoom
- `Shift + Left drag`: pan
- Box mode + left drag: create box prompt

## Keyboard

- `Enter`: confirm current instance
- `S`: save current image annotations
- `Space` / `Right`: next image
- `Left`: previous image
- `B`: toggle box mode
- `E`: toggle edit-mask mode
- `T`: revert current mask to SAM prediction
- `Backspace`: undo last confirmed instance
- `U`: undo last point
- `R`: reset current points/mask
- `[` / `]`: brush radius - / +
- `N` / `P` / `1~9`: switch class

## UI Actions

- `Flag` button: toggle flag for current image
- `Overview` button: open thumbnail wall and jump to image

## Notes

- `Enter` confirms in memory only
- `S` writes outputs to disk
- Confirmed instances restore from autosave when revisiting image
- Flag restore is disabled by default; use `--restore-flags` to enable
