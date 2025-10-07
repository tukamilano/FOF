input_token = ['a', 'b', 'c', '(', ')', '→', '∧', '∨', '⊥']

output = ["assumption", "intro", "split", "left", "right", "add_dn"]
for i in range(3):
    output.append(f"apply {i}")
    output.append(f"destruct {i}")
for i in range(3):
    for j in range(3):
        if i != j:
            output.append(f"specialize {i} {j}")


