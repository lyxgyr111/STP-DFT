import random
text_templates = [
    "[TEXT_START] The initial reading for the sensor was {val}. [TEXT_END]",
    "[TEXT_START] After a period of stabilization, the value settled at {val}. [TEXT_END]",
    "[TEXT_START] A sudden spike was observed, peaking at {val}. [TEXT_END]",
    "[TEXT_START] System log: process complete. The final state is {val}. [TEXT_END]",
    "[TEXT_START] Patient update: the metric is now {val}. [TEXT_END]",
]
modalities = {
    'TEMP': lambda: f"{random.uniform(10.0, 99.9):.1f}",
    'PRES': lambda: f"{random.uniform(1.0, 10.0):.1f}",
    'HR': lambda: str(random.randint(60, 140)),
    'SPO2': lambda: str(random.randint(90, 100)),
    'STATUS': lambda: str(random.randint(0, 1)),
}
NUM_LINES = 5000  
output_filename = 'data/input_heterogeneous.txt'
with open(output_filename, 'w', encoding='utf-8') as f:
    for _ in range(NUM_LINES):
        num_segments = random.randint(1, 5)
        line_segments = []
        for _ in range(num_segments):
            modality_key = random.choice(list(modalities.keys()))
            value_generator = modalities[modality_key]
            modality_value = value_generator()
            modality_str = f"[{modality_key}_START] {modality_value} [{modality_key}_END]"
            template = random.choice(text_templates)
            segment = template.format(val=modality_str)
            line_segments.append(segment)
        f.write(" ".join(line_segments) + "\n")
print(f"Successfully generated {NUM_LINES} lines of heterogeneous data into {output_filename}")