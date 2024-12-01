0 lda #104
1 sta 51
2 lda #101
3 sta 52
4 lda #108
5 sta 53
6 lda #108
7 sta 54
8 lda #111
9 sta 55
10 lda #0
11 sta 56
12 lda #0
13 sta 57
14 lda #0
15 sta 58
16 lda #0
17 sta 59
18 lda #0
19 sta 60
20 lda #0
21 sta 61
22 lda #0
23 sta 62
24 lda #3
25 sta 66
26 lda #51
27 sta 67
28 lda #57
29 sta 63
30 lda 66
31 sta 65
32 lda #32
33 sta 68
34 lda #6
35 add 68
36 sta 68
37 jmp 39
38 stp
39 lda (67)
40 jze 50
41 add 65
42 sta (63)
43 lda #1
44 add 67
45 sta 67
46 lda #1
47 add 63
48 sta 63
49 jmp 39
50 jmp (68)
51 0
52 0
53 0
54 0
55 0
56 0
57 0
58 0
59 0
60 0
61 0
62 0
63 0
64 0
65 0
66 0
67 0
68 0
