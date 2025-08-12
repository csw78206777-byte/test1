// 食物列表
const foodList = [
    '拉面', '盖饭', '麻辣烫', '沙拉', '汉堡',
    '炒饭', '饺子', '包子', '面条', '寿司',
    '披萨', '三明治', '粥', '煎饼', '烤肉',
    '火锅', '炸鸡', '牛排', '意面', '烧烤',
    '小笼包', '煲仔饭', '兰州拉面', '黄焖鸡', '沙县小吃',
    '麦当劳', '肯德基', '必胜客', '海底捞', '真功夫'
];

// 获取DOM元素
const selectBtn = document.getElementById('selectBtn');
const result = document.getElementById('result');

// 点击事件处理
selectBtn.addEventListener('click', function() {
    // 添加旋转动画
    selectBtn.classList.add('spinning');
    
    // 延迟显示结果，增加悬念
    setTimeout(() => {
        // 随机选择食物
        const randomIndex = Math.floor(Math.random() * foodList.length);
        const selectedFood = foodList[randomIndex];
        
        // 显示结果
        result.textContent = selectedFood;
        result.classList.add('show', 'animate');
        
        // 移除按钮旋转动画
        selectBtn.classList.remove('spinning');
        
        // 移除结果动画类，为下次点击做准备
        setTimeout(() => {
            result.classList.remove('animate');
        }, 600);
        
    }, 500);
});

// 添加键盘事件支持（按空格键也可以选择）
document.addEventListener('keydown', function(event) {
    if (event.code === 'Space') {
        event.preventDefault();
        selectBtn.click();
    }
});

// 页面加载完成后的欢迎动画
window.addEventListener('load', function() {
    const container = document.querySelector('.container');
    container.style.opacity = '0';
    container.style.transform = 'translateY(50px)';
    
    setTimeout(() => {
        container.style.transition = 'all 0.8s ease';
        container.style.opacity = '1';
        container.style.transform = 'translateY(0)';
    }, 100);
});