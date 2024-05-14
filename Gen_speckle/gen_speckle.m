function I=gen_speckle(wid,hei,r)
I=zeros(wid,hei);
spacing=2*r/sqrt(2/3.14159);
rows_space=fix(wid/spacing);
cols_space=fix(hei/spacing);
X_grid=zeros(rows_space+1,cols_space+1);
Y_grid=zeros(rows_space+1,cols_space+1);
xmin = 0.5 * (wid - rows_space * spacing);
ymin = 0.5 * (hei - cols_space * spacing);
for ii=1:rows_space+1
    for jj=1:cols_space+1
        X_grid(ii,jj)=xmin + (ii-1) * spacing;
        Y_grid(ii,jj)=ymin + (jj-1) * spacing;
    end
end
limit = 0.5 * 0.5 * spacing;
X_grid = X_grid + limit * 2 * (rand(rows_space+1,cols_space+1) - 0.5);
Y_grid = Y_grid + limit * 2 * (rand(rows_space+1,cols_space+1) - 0.5);
for ii=1:size(X_grid,1)
    for jj=1:size(X_grid,2)
        delta_x1=max(1,round(X_grid(ii,jj)-r-2));
        delta_x2=min(wid,round(X_grid(ii,jj)+r+2));
        delta_y1=max(1,round(Y_grid(ii,jj)-r-2));
        delta_y2=min(hei,round(Y_grid(ii,jj)+r+2));
        for pp=delta_x1:delta_x2
            for qq=delta_y1:delta_y2
                if sqrt((pp-X_grid(ii,jj))^2+(qq-Y_grid(ii,jj))^2)<r
                    I(pp,qq)=1;
                end
            end
        end
        % dis_now=sqrt((X-X_grid(ii,jj)).^2+(Y-Y_grid(ii,jj)).^2);
        % I(dis_now<r)=1;
    end
end
end