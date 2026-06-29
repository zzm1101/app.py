// static/js/feature_spc_batch.js
let activeTabProduct = null;
let currentCharts = {};

$(function() {
    $('button[data-bs-toggle="tab"]').on('shown.bs.tab', function(e) {
        const targetId = $(e.target).attr('data-bs-target');
        if (targetId === '#tab_all') return;
        const product = $(e.target).data('product');
        if (!product) return;
        activeTabProduct = product;
        loadProductData(product);
    });

    $('#refreshAllBtn').click(function() {
        if (activeTabProduct) {
            loadProductData(activeTabProduct);
        } else {
            const activeTab = $('.nav-tabs .nav-link.active');
            if (activeTab && activeTab.data('product')) {
                activeTabProduct = activeTab.data('product');
                loadProductData(activeTabProduct);
            }
        }
    });

    $('#resetRulesBtn').click(function() {
        $('.rule-check').prop('checked', true);
    });
});

function loadProductData(productItem) {
    const tabIdx = getTabIndex(productItem);
    const contentDiv = $(`#content_${tabIdx}`);
    const loadingDiv = $(`#loading_${tabIdx}`);

    loadingDiv.show();
    contentDiv.empty();

    const rules = $('.rule-check:checked').map(function() { return this.value; }).get().join(',');
    const startDate = $('#globalStartDate').val();
    const endDate = $('#globalEndDate').val();

    $.getJSON('/feature-monitor/api/batch_data', {
        product_item: productItem,
        start_date: startDate,
        end_date: endDate,
        rules: rules
    })
    .done(function(res) {
        renderProductCharts(contentDiv, res);
    })
    .fail(function(xhr) {
        let errMsg = '数据加载失败';
        try {
            const resp = xhr.responseJSON;
            if (resp && resp.error) errMsg = resp.error;
        } catch(e) {}
        contentDiv.html(`<div class="alert alert-danger">${errMsg}</div>`);
    })
    .always(function() {
        loadingDiv.hide();
    });
}

function renderProductCharts(container, data) {
    if (!data.ctqs || data.ctqs.length === 0) {
        container.html('<div class="alert alert-warning">该品项下没有找到可监控的影响因素数据，请确认是否已上传特征数据（至少需要2个批次）。</div>');
        return;
    }

    let html = '';
    for (let ctq of data.ctqs) {
        html += `
            <div class="ctq-card" data-ctq-id="${ctq.ctq_id}">
                <div class="ctq-header">
                    <i class="fa fa-cogs me-2"></i> ${escapeHtml(ctq.ctq_name)}
                </div>
                <div class="feature-grid" id="grid_${ctq.ctq_id}"></div>
            </div>
        `;
    }
    container.html(html);

    for (let ctq of data.ctqs) {
        const grid = $(`#grid_${ctq.ctq_id}`);
        for (let feat of ctq.features) {
            const safeFeatId = safeId(feat.feature_name);
            const featureCard = $(`
                <div class="feature-card" data-feature="${safeFeatId}">
                    <div class="card-header">
                        <i class="fa fa-bar-chart me-1"></i> ${escapeHtml(feat.feature_name)}
                    </div>
                    <div class="card-body">
                        <div id="i_${data.product_item}_${ctq.ctq_id}_${safeFeatId}" class="chart-container"></div>
                    </div>
                    <div class="advice-text" id="advice_${data.product_item}_${ctq.ctq_id}_${safeFeatId}"></div>
                </div>
            `);
            grid.append(featureCard);
        }
    }

    // 绘制所有图表
    for (let ctq of data.ctqs) {
        for (let feat of ctq.features) {
            const safeFeatId = safeId(feat.feature_name);
            const chartId = `i_${data.product_item}_${ctq.ctq_id}_${safeFeatId}`;
            const adviceId = `advice_${data.product_item}_${ctq.ctq_id}_${safeFeatId}`;
            setTimeout(() => {
                drawIndividualChart(chartId, feat);
                generateAdviceText(adviceId, feat);
            }, 50);
        }
    }
}

function drawIndividualChart(containerId, data) {
    const dom = document.getElementById(containerId);
    if (!dom) {
        console.warn(`容器不存在: ${containerId}`);
        return;
    }
    if (!data.xbar || data.xbar.length === 0) {
        dom.innerHTML = '<div class="alert alert-warning small">无有效数据</div>';
        return;
    }
    if (currentCharts[containerId]) {
        currentCharts[containerId].dispose();
        delete currentCharts[containerId];
    }
    const chart = echarts.init(dom);
    const labels = data.labels || data.xbar.map((_, i) => i+1);

    // 横坐标优化
    const labelCount = labels.length;
    let interval = 0;
    if (labelCount > 30) {
        interval = Math.floor(labelCount / 30);
    } else if (labelCount > 15) {
        interval = Math.floor(labelCount / 15);
    }

        chart.setOption({
            tooltip: { trigger: 'axis' },
            legend: { data: ['实测值', 'UCL', 'CL', 'LCL'], left: 'left', itemWidth: 20 },
            toolbox: {
                feature: {
                    dataZoom: { yAxisIndex: 'none' },
                    restore: {},
                    saveAsImage: {}
                }
            },
            grid: { containLabel: true, left: '5%', right: '5%', bottom: '8%' },
            dataZoom: [
                { type: 'slider', start: 0, end: 100, bottom: 5 },
                { type: 'inside', start: 0, end: 100 }
            ],
            xAxis: {
                type: 'category',
                data: labels,
                axisLabel: {
                    rotate: 45,
                    interval: interval,
                    fontSize: 10,
                    margin: 15
                }
            },
            yAxis: { type: 'value', name: '特征值', scale: true },
            series: [
                {
                    name: '实测值',
                    type: 'line',
                    data: data.xbar,
                    symbol: 'circle',
                    symbolSize: 5,
                    lineStyle: { color: '#6366f1' },
                    markPoint: {
                        symbol: 'circle', symbolSize: 6, data: (data.alarm_x||[]).map(i => ({ coord: [i, data.xbar[i]], name: '异常', itemStyle: { color: '#ef4444' } }))
                    }
                },
            { name: 'UCL', type: 'line', data: Array(labels.length).fill(data.ucl_x), lineStyle: { type: 'dashed', color: '#ef4444' }, symbol: 'none' },
            { name: 'CL', type: 'line', data: Array(labels.length).fill(data.cl_x), lineStyle: { type: 'dashed', color: '#10b981' }, symbol: 'none' },
            { name: 'LCL', type: 'line', data: Array(labels.length).fill(data.lcl_x), lineStyle: { type: 'dashed', color: '#ef4444' }, symbol: 'none' }
        ]
    });
    currentCharts[containerId] = chart;
    window.addEventListener('resize', () => chart.resize());
}

function generateAdviceText(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;
    let advice = '';
    if (data.rules_violations && data.rules_violations.length > 0) {
        advice += `<i class="fa fa-exclamation-triangle text-warning"></i> 检测到 ${data.rules_violations.length} 个异常点。`;
    }
    if (data.cpk !== null && data.cpk !== undefined && data.cpk < 1.33) {
        advice += ` 过程能力 CpK = ${data.cpk.toFixed(2)} < 1.33，波动较大。`;
    }
    if (!advice) advice = '过程受控，无异常。';
    container.innerHTML = advice;
}

function getTabIndex(productItem) {
    const tabs = $('.nav-tabs button[data-product]');
    for (let i = 0; i < tabs.length; i++) {
        if ($(tabs[i]).data('product') === productItem) {
            return i + 1;
        }
    }
    return 0;
}

function safeId(str) {
    return str.replace(/[^a-zA-Z0-9\u4e00-\u9fa5]/g, '_').substring(0, 50);
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/[&<>]/g, function(m) {
        if (m === '&') return '&amp;';
        if (m === '<') return '&lt;';
        if (m === '>') return '&gt;';
        return m;
    });
}