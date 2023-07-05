struct obj(){
    int x, y;//位置
    int width, height;//尺寸
    vector<float>outlook(max(width,height), 0);//外观特征向量
};
vector<float>cal_outlook(vector<vector<int>>image, obj a)//特征向量计算
{
    if(a.width>a.height){
        vector<float>outlook(a.width, 0);
        for(int i=0;i<a.width;i++){
            for(int j=0;j<a.height;j++){
                outlook[i]+=image[a.x-a.width/2+i][a.y-a.height/2+j];
            }
            outlook[i]/=a.height;
        }
        return outlook;
    }
    else {
        vector<float>outlook(a.height, 0);
        for(int j=0;j<a.height;j++){
            for(int i=0;i<a.width;i++){
                outlook[j]+=image[a.x-a.width/2+i][a.y-a.height/2+j];
            }
            outlook[j]/=a.width;
        }
        return outlook;
    }
}
vector<float>resize(vector<float>a, int n){
    int m=a.size();
    vector<float>ans(n,0);
    for(int i=0;i<n;i++){
        int k=i*m/n;
        float b=1.0*i*m/n-k;
        ans[i]=a[k]*(1-b)+a[k+1]*b;
    } 
}
float match(obj a,obj b)//两个物体相似度计算
{
    float sim_size=a.width*b.height/(a.height*b.width);//尺寸相似度
    if(sim_size<0.5||sim_size>2.0)return 0.0;
    float sim_dist=abs(a.x-b.x)+abs(a.y-b.y);//距离相似度
    if(sim_dist>(a.width+a.height)&&sim_dist>(b.width+b.height))return 0.0;
    float sim_look=0.0;//外观相似度
    int n=max(a.outlook.size(),b.outlook.size());
    if(a.outlook.size()<b.outlook.size())a.outlook=resize(a.outlook, n);//短向量拉伸到长向量的大小
    else b.outlook=resize(b.outlook, n);
    float da=sqrt(sum(a.outlook)),db=sqrt(sum(b.outlook));
    for(int i=0;i<n;i++)sim_look+=a.outlook[i]*b.outlook[i];
    return sim_look/(da*db);
}
