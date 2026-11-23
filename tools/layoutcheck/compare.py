import sys
def load_reflex(p):
    out={}
    for l in open(p, encoding='utf-8'):
        if l.startswith('//') or not l.strip(): continue
        f=l.rstrip('\n').split('\t'); out[f[0]]=f[1:5]
    return out
def load_mac(p, iso):
    out={}
    for l in open(p, encoding='utf-8'):
        f=l.rstrip('\n').split('\t')
        name=f[0]
        if iso: name={'Sect':'Backquote','Grave':'IntlBackslash'}.get(name,name)
        else: name={'Grave':'Backquote'}.get(name,name)
        if name=='Sect' and not iso: continue
        out[name]=f[1:5]
    return out
# Spacing forms macOS returns for dead keys (dead key + space) vs. the plain marks in the tables.
SPACING = {'ˆ': '^', '˜': '~'}
def norm(c):
    return {'minus':'-','space':' '}.get(c,c)
def norm_mac(c):
    return 'dead:' + SPACING.get(c[5:], c[5:]) if c.startswith('dead:') else c
# Usage: compare.py <reflex table> <macOS dump> <iso|ansi> [swap]
# "swap": the core swaps Backquote/IntlBackslash for this layout on ISO hardware.
reflex, mac = load_reflex(sys.argv[1]), load_mac(sys.argv[2], sys.argv[3]=='iso')
if len(sys.argv) > 4 and sys.argv[4] == 'swap':
    reflex['Backquote'], reflex['IntlBackslash'] = reflex['IntlBackslash'], reflex['Backquote']
bad=0
for code, cells in reflex.items():
    for layer,(r,m) in enumerate(zip(cells, mac.get(code,['?']*4))):
        if r=='-': continue
        if norm(r)!=norm_mac(m):
            bad+=1; print(f"{code} layer{layer}: reflex={r!r} mac={m!r}")
print("mismatches:", bad, "checked keys:", len(reflex))
sys.exit(1 if bad else 0)
