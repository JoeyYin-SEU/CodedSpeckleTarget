function I=gen_ring_speckle(wid,hei,r,code_bit)
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
X=repmat((1:wid)',[1,hei]);
Y=repmat((1:hei),[wid,1]);
X_list=[];
Y_list=[];
for ii=1:rows_space+1
    for jj=1:cols_space+1
        theta_now=atan2(-X_grid(ii,jj)+wid/2,-Y_grid(ii,jj)+hei/2);
        if theta_now<0
            theta_now=2*pi+theta_now;
        end
        theta_nowa=theta_now/2/pi*360;
        dis2_0=abs(X_grid(ii,jj)-wid/2);
        dis2_1=abs(tan(theta_now)*Y_grid(ii,jj)+X_grid(ii,jj)-wid/2+tan(theta_now)*hei/2)/sqrt(1+tan(theta_now)^2);
        if theta_nowa>0 &&theta_nowa<360/code_bit
            X_list=[X_list,X_grid(ii,jj)];
            Y_list=[Y_list,Y_grid(ii,jj)];
        end
    end
end
Centre_matrix=[1,0,wid/2;0,1,hei/2;0,0,1];
Centre_matrix_inv=[1,0,-wid/2;0,1,-hei/2;0,0,1];
X_list=repmat(X_list,[code_bit,1]);
Y_list=repmat(Y_list,[code_bit,1]);
for ii=2:code_bit
    for jj=1:size(X_list,2)
        thata_r=2*pi/code_bit*(ii-1);
        R_matrix=[cos(thata_r),sin(thata_r),0;-sin(thata_r),cos(thata_r),0;0,0,1];
        XY=[X_list(1,jj),Y_list(1,jj),1];
        XY_new=Centre_matrix*R_matrix*Centre_matrix_inv*XY';
        X_list(ii,jj)=XY_new(1);
        Y_list(ii,jj)=XY_new(2);
    end
end
for ii=1:size(X_list,1)
    for jj=1:size(X_list,2)

        delta_x1=max(1,round(X_list(ii,jj)-r-2));
        delta_x2=min(wid,round(X_list(ii,jj)+r+2));
        delta_y1=max(1,round(Y_list(ii,jj)-r-2));
        delta_y2=min(hei,round(Y_list(ii,jj)+r+2));
        for pp=delta_x1:delta_x2
            for qq=delta_y1:delta_y2
                if sqrt((pp-X_list(ii,jj))^2+(qq-Y_list(ii,jj))^2)<r
                    I(pp,qq)=1;
                end
            end
        end
        % dis_now=sqrt((X-X_list(ii,jj)).^2+(Y-Y_list(ii,jj)).^2);
        % I(dis_now<r)=1;
    end
end
end