algebraic3d
solid star = sphere(0, 0, 0; 0.35);
solid atm = sphere(0, 0, 0; 10);

solid air = atm and not star;

tlo star -col=[1,0,0];
tlo air -col=[0,0,1] -transparent;
