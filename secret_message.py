import base64
encrypted = "HlUaFk0OCF0AT0VURVUOEUsMGQlfSEINCh4FBk8KGEtUSF9OQhcaF0sIAEsXT0lOQhcPBUEfGV1U SF9OQhsHAFwICUcRBABJSVJOAk0FBEsFDQgLCwZOQxRNSlsdBAoNDhcNRAJNSlwSCgcHEQFOQxRN Sl0SDgBJSVJOBUECSg5JSEIZDBxIRFM="
my_eyes = str.encode("eric.mm.shen")
decoded = base64.b64decode(encrypted)
decrypted = ""
for i in range(0, len(decoded)):
    decrypted += chr((ord(my_eyes[i % len(my_eyes)]) ^ ord(decoded[i])))
print(decrypted)
