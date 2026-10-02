# Data dictionary

`campus_events.csv` is a project-created synthetic dataset. One row represents one proposed campus event. Duration is measured in hours. `equipment_needs` records the primary equipment requirement, not an exhaustive inventory.

| Column | CSV data type | Meaning | Valid values or range | Role | Classification relevance |
|---|---|---|---|---|---|
| `participants` | integer | Expected attendance | Positive whole number | Input | Audience size can vary by event type, with overlap. |
| `duration` | decimal number | Planned duration in hours | Positive finite number | Input | Event length provides context alongside other features. |
| `venue` | text category | Planned campus space | Auditorium; Lecture Hall; Conference Room; Multipurpose Hall; Gymnasium; Open Field | Input | Venue suitability contributes context but does not determine the class alone. |
| `equipment_needs` | text category | Primary equipment need | Projector; Computers; Microphones; Sound System; Sports Gear; No Equipment | Input | Equipment can indicate activity, and several categories share equipment. |
| `event_type` | text category | Proposed event category | Academic; Sports; Cultural; Organizational | Target | The four classes the models learn to predict. |

Rows contain no personal details, identifiers, or features copied from the target. The same CSV can be read directly in KNIME with a CSV Reader node.
