# Advanced Parameters Quick Reference

## Parameter Categories & Defaults

### Rest & Travel
| Parameter | Default | Range | Example |
|-----------|---------|-------|---------|
| Rest days home | 3 | 0-7 | `2` for short rest |
| Rest days away | 3 | 0-7 | `1` for back-to-back |
| Travel km away | 500 | 0-3000 | `2800` for cross-country |
| Timezone change away | 0 | -3 to 3 | `3` for 3hr east travel |

### Back-to-Back & Momentum
| Parameter | Default | Options | Example |
|-----------|---------|---------|---------|
| Home B2B | No | y/n | `y` if playing 2nd of B2B |
| Away B2B | No | y/n | `y` if playing 2nd of B2B |
| Home win streak | 0 | -10 to 10 | `5` for 5-game streak |
| Away win streak | 0 | -10 to 10 | `-3` for 3-game losing streak |

### Venue & Environmental
| Parameter | Default | Options | Note |
|-----------|---------|---------|------|
| High altitude | No | y/n | Denver, Salt Lake City |

### Head-to-Head & Season Context
| Parameter | Default | Range | Example |
|-----------|---------|-------|---------|
| Home H2H win rate | 0.5 | 0.0-1.0 | `0.75` = 75% home win rate |
| H2H meetings season | 0 | 0-4 | `2` if they've played twice |
| Month | 3 (March) | 1-12 | `4` for April (late season) |
| Day of week | 2 (Tuesday) | 0-6 | `6` for Sunday games |
| Is weekend | No | y/n | `y` for Saturday/Sunday |

### Playoff-Specific
| Parameter | Default | Range | Example |
|-----------|---------|-------|---------|
| Playoff round | 1 | 1-4 | `4` for Finals |
| Series game num | 0 | 1-7 | `6` for Game 6 (deciding) |
| Must-win game | No | y/n | `y` for elimination game |

### Performance Trends
| Parameter | Default | Range | Example |
|-----------|---------|-------|---------|
| Home pts trend | 0 | -10 to 10 | `3` if scoring 3 more PPG |
| Away pts trend | 0 | -10 to 10 | `-2` if scoring 2 fewer PPG |

---

## Example Scenarios

### Scenario 1: Back-to-Back Away Game (Low Rest)
```
Rest days away: 1
Travel km away: 2500
Away team back-to-back: y
```

### Scenario 2: Altitude Advantage (Denver Home Game)
```
High altitude: y
Rest days home: 3
Rest days away: 1
```

### Scenario 3: Finals Game 7 (Must-Win)
```
Playoff round: 4
Series game number: 7
Must-win game: y
Month: 6
```

### Scenario 4: Playoff First Round, Series Game 5
```
Playoff round: 1
Series game number: 5
Home win streak: 3
Away win streak: -2
```

### Scenario 5: Conference Rival (High H2H History)
```
Home H2H win rate: 0.65
H2H meetings this season: 2
Month: 3
```

---

## Tips

- **Skip parameters** by pressing Enter (uses defaults)
- **Mix and match** — combine any parameters you want
- **Default values** are reasonable NBA averages
- **Validation** — script checks ranges and types automatically
- **Performance trends** are subtle adjustments (±10 PPG range)
