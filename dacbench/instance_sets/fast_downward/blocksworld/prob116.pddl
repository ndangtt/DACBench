

(define (problem BW-rand-21)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 )
(:init
(arm-empty)
(on b1 b9)
(on b2 b19)
(on b3 b5)
(on b4 b16)
(on-table b5)
(on-table b6)
(on b7 b15)
(on b8 b4)
(on-table b9)
(on-table b10)
(on b11 b7)
(on-table b12)
(on-table b13)
(on b14 b10)
(on b15 b12)
(on b16 b11)
(on b17 b21)
(on b18 b13)
(on-table b19)
(on b20 b14)
(on b21 b6)
(clear b1)
(clear b2)
(clear b3)
(clear b8)
(clear b17)
(clear b18)
(clear b20)
)
(:goal
(and
(on b1 b7)
(on b2 b4)
(on b3 b16)
(on b4 b20)
(on b7 b9)
(on b9 b6)
(on b11 b18)
(on b12 b5)
(on b13 b19)
(on b14 b12)
(on b15 b14)
(on b16 b15)
(on b18 b10)
(on b19 b8)
(on b20 b13)
(on b21 b17))
)
)


