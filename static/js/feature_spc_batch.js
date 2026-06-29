// feature_spc_batch.js —— 影响因素控制图（按品项）
// 修复 $.when 参数解析问题，支持规格限与固定控制限

let activeTabProduct = null;
let currentCharts = {};
let allLimits = {};                          // 界限配置缓存
let allChartData = {};                       // 原始图表数据缓存

$(function() {
    // Tab 切换事件
    $('button[data-bs-toggle="tab"]').on('shown.bs.tab', function(e) {
        const targetId = $(e.target).attr('data-bs-target');
        if (targetId === '#tab_all' || targetId === '#tab_limits') return;
        const product = $(e.target).data('product');
        if (!product) return;
        activeTabProduct = product;
        loadProductData(product);
    });

    // 刷新按钮
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

    // 规则全选
    $('#resetRulesBtn').click(function() {
        $('.rule-check').prop('checked', true);
    });

    // ---------- 规格限设置面板逻辑 ----------
    const $limitsProductSelect = $('#limitsProductSelect');
    const $limitsTableContainer = $('#limitsTableContainer');
    const $limitsAlert = $('#limitsAlert');

    $('#loadLimitsBtn').click(async function() {
        const product = $limitsProductSelect.val();
        if (!product) {
            $limitsAlert.show().removeClass('alert-success').addClass('alert-warning').text('请先选择品项');
            return;
        }
        try {
            const batchResp = await $.getJSON('/feature-monitor/api/batch_data', { product_item: product });
            const limitsResp = await $.getJSON('/feature-monitor/api/feature_limits', { product_item: product });
            const limits = limitsResp.limits || {};

            let html = '<table class="table table-bordered table-sm"><thead><tr>';
            html += '<th>CTQ</th><th>特征</th><th>USL</th><th>LSL</th><th>UCL(固定)</th><th>CL(固定)</th><th>LCL(固定)</th><th>显示规格</th><th>固定控制限</th></tr></thead><tbody>';

            if (batchResp.ctqs && batchResp.ctqs.length) {
                batchResp.ctqs.forEach(ctq => {
                    ctq.features.forEach(feat => {
                        const key = `${ctq.ctq_id}_${feat.feature_name}`;
                        const cfg = limits[key] || {};
                        html += `<tr>
                            <td>${escapeHtml(ctq.ctq_name)}</td>
                            <td>${escapeHtml(feat.feature_name)}</td>
                            <td><input type="number" step="any" class="form-control form-control-sm spec-usl" data-key="${key}" value="${cfg.usl || ''}"></td>
                            <td><input type="number" step="any" class="form-control form-control-sm spec-lsl" data-key="${key}" value="${cfg.lsl || ''}"></td>
                            <td><input type="number" step="any" class="form-control form-control-sm fixed-ucl" data-key="${key}" value="${cfg.ucl || ''}"></td>
                            <td><input type="number" step="any" class="form-control form-control-sm fixed-cl" data-key="${key}" value="${cfg.cl || ''}"></td>
                            <td><input type="number" step="any" class="form-control form-control-sm fixed-lcl" data-key="${key}" value="${cfg.lcl || ''}"></td>
                            <td class="text-center"><input type="checkbox" class="form-check-input spec-show" data-key="${key}" ${cfg.show_spec ? 'checked' : ''}></td>
                            <td class="text-center"><input type="checkbox" class="form-check-input fixed-use" data-key="${key}" ${cfg.use_fixed_control ? 'checked' : ''}></td>
                        </tr>`;
                    });
                });
            } else {
                html += '<tr><td colspan="9" class="text-center text-muted">该品项暂无特征数据</td></tr>';
            }
            html += '</tbody></table>';
            $limitsTableContainer.html(html);
            $limitsAlert.hide();
        } catch (err) {
            $limitsAlert.show().removeClass('alert-success alert-warning').addClass('alert-danger').text('加载失败：' + err.message);
        }
    });

    $('#saveLimitsBtn').click(async function() {
        const product = $limitsProductSelect.val();
        if (!product) {
            alert('请先选择品项');
            return;
        }
        const rows = $limitsTableContainer.find('tbody tr');
        const payload = { product_item: product, limits: [] };

        rows.each(function() {
            const $row = $(this);
            const keyInput = $row.find('[data-key]');
            if (!keyInput.length) return;
            const key = keyInput.attr('data-key');
            const [ctqId, featureName] = key.split('_');
            const usl = $row.find('.spec-usl').val();
            const lsl = $row.find('.spec-lsl').val();
            const ucl = $row.find('.fixed-ucl').val();
            const cl  = $row.find('.fixed-cl').val();
            const lcl = $row.find('.fixed-lcl').val();
            const showSpec = $row.find('.spec-show').is(':checked');
            const useFixed = $row.find('.fixed-use').is(':checked');

            payload.limits.push({
                ctq_id: parseInt(ctqId),
                feature_name: featureName,
                usl: usl === '' ? null : parseFloat(usl),
                lsl: lsl === '' ? null : parseFloat(lsl),
                ucl: ucl === '' ? null : parseFloat(ucl),
                cl:  cl === '' ? null : parseFloat(cl),
                lcl: lcl === '' ? null : parseFloat(lcl),
                show_spec: showSpec,
                use_fixed_control: useFixed
            });
        });

        try {
            await $.ajax({
                url: '/feature-monitor/api/feature_limits',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(payload)
            });
            $limitsAlert.show().removeClass('alert-warning alert-danger').addClass('alert-success').text('保存成功！');
        } catch (err) {
            $limitsAlert.show().removeClass('alert-success alert-warning').addClass('alert-danger').text('保存失败：' + err.message);
        }
    });
});

// ==================== 数据加载 ====================
function loadProductData(productItem) {
    // 销毁已有图表
    Object.values(currentCharts).forEach(chart => chart.dispose());
    currentCharts = {};
    allChartData = {};

    const tabIdx = getTabIndex(productItem);
    const contentDiv = $(`#content_${tabIdx}`);
    const loadingDiv = $(`#loading_${tabIdx}`);

    if (!contentDiv.length) {
        console.warn(`未找到品项容器 content_${tabIdx}`);
        return;
    }

    loadingDiv.show();
    contentDiv.empty();

    const rules = $('.rule-check:checked').map(function() { return this.value; }).get().join(',');
    const startDate = $('#globalStartDate').val();
    const endDate = $('#globalEndDate').val();

    const batchReq = $.getJSON('/feature-monitor/api/batch_data', {
        product_item: productItem,
        start_date: startDate,
        end_date: endDate,
        rules: rules
    });
    const limitsReq = $.getJSON('/feature-monitor/api/feature_limits', { product_item: productItem });

    // 修正：正确解析 $.when 返回的数组参数
    $.when(batchReq, limitsReq)
        .done(function(batchResult, limitsResult) {
            // batchResult 结构： [data, statusText, jqXHR]
            const batchData = batchResult[0];
            const limitsData = limitsResult[0];
            allLimits = (limitsData && limitsData.limits) ? limitsData.limits : {};
            renderProductCharts(contentDiv, batchData);
        })
        .fail(function(xhr, textStatus, errorThrown) {
            let errMsg = '数据加载失败';
            if (xhr && xhr.responseJSON && xhr.responseJSON.error) {
                errMsg = xhr.responseJSON.error;
            }
            contentDiv.html(`<div class="alert alert-danger">${errMsg}</div>`);
        })
        .always(function() {
            loadingDiv.hide();
        });
}

// ==================== 图表渲染 ====================
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
            const chartId = `i_${data.product_item}_${ctq.ctq_id}_${safeFeatId}`;
            allChartData[chartId] = feat;

            const limitKey = `${ctq.ctq_id}_${feat.feature_name}`;
            const limitConfig = allLimits[limitKey] || {};

            const featureCard = $(`
                <div class="feature-card" data-feature="${safeFeatId}">
                    <div class="card-header">
                        <i class="fa fa-bar-chart me-1"></i> ${escapeHtml(feat.feature_name)}
                        <div>
                            <div class="form-check form-check-inline form-switch mb-0">
                                <input class="form-check-input spec-switch" type="checkbox" data-chart-id="${chartId}" data-limit-key="${limitKey}" ${limitConfig.show_spec ? 'checked' : ''}>
                                <label class="form-check-label switch-label">显示规格</label>
                            </div>
                            <div class="form-check form-check-inline form-switch mb-0">
                                <input class="form-check-input fixed-ctrl-switch" type="checkbox" data-chart-id="${chartId}" data-limit-key="${limitKey}" ${limitConfig.use_fixed_control ? 'checked' : ''}>
                                <label class="form-check-label switch-label">固定控制限</label>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="${chartId}" class="chart-container"></div>
                    </div>
                    <div class="advice-text" id="advice_${chartId}"></div>
                </div>
            `);
            grid.append(featureCard);
        }
    }

    bindLimitSwitches();

    // 绘制所有图表
    for (let ctq of data.ctqs) {
        for (let feat of ctq.features) {
            const safeFeatId = safeId(feat.feature_name);
            const chartId = `i_${data.product_item}_${ctq.ctq_id}_${safeFeatId}`;
            const limitKey = `${ctq.ctq_id}_${feat.feature_name}`;
            const limitConfig = allLimits[limitKey] || {};
            setTimeout(() => {
                drawIndividualChart(chartId, feat, limitConfig);
                generateAdviceText(`advice_${chartId}`, feat);
            }, 50);
        }
    }
}

// 开关事件处理
function bindLimitSwitches() {
    $('.spec-switch, .fixed-ctrl-switch').off('change').on('change', function() {
        const $switch = $(this);
        const chartId = $switch.data('chart-id');
        const limitKey = $switch.data('limit-key');
        const isSpec = $switch.hasClass('spec-switch');
        const field = isSpec ? 'show_spec' : 'use_fixed_control';

        if (!allLimits[limitKey]) {
            allLimits[limitKey] = {};
        }
        allLimits[limitKey][field] = $switch.is(':checked');

        const featData = allChartData[chartId];
        if (featData) {
            drawIndividualChart(chartId, featData, allLimits[limitKey]);
        }
    });
}

// ==================== 单图绘制 ====================
function drawIndividualChart(containerId, data, limitConfig) {
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
    currentCharts[containerId] = chart;

    const labels = data.labels || data.xbar.map((_, i) => i + 1);
    const labelCount = labels.length;
    let interval = 0;
    if (labelCount > 30) interval = Math.floor(labelCount / 30);
    else if (labelCount > 15) interval = Math.floor(labelCount / 15);

    const series = [];

    series.push({
        name: '实测值',
        type: 'line',
        data: data.xbar,
        symbol: 'circle',
        symbolSize: 5,
        lineStyle: { color: '#6366f1' },
        markPoint: {
            symbol: 'circle',
            symbolSize: 6,
            data: (data.alarm_x || []).map(i => ({
                coord: [i, data.xbar[i]],
                name: '异常',
                itemStyle: { color: '#ef4444' }
            }))
        }
    });

    const useFixed = limitConfig && limitConfig.use_fixed_control;
    const uclVal = useFixed && limitConfig.ucl != null ? limitConfig.ucl : data.ucl_x;
    const clVal  = useFixed && limitConfig.cl  != null ? limitConfig.cl  : data.cl_x;
    const lclVal = useFixed && limitConfig.lcl != null ? limitConfig.lcl : data.lcl_x;

    series.push({
        name: 'UCL' + (useFixed ? '(固定)' : ''),
        type: 'line',
        data: Array(labels.length).fill(uclVal),
        lineStyle: { type: 'dashed', color: '#ef4444' },
        symbol: 'none'
    });
    series.push({
        name: 'CL' + (useFixed ? '(固定)' : ''),
        type: 'line',
        data: Array(labels.length).fill(clVal),
        lineStyle: { type: 'dashed', color: '#10b981' },
        symbol: 'none'
    });
    series.push({
        name: 'LCL' + (useFixed ? '(固定)' : ''),
        type: 'line',
        data: Array(labels.length).fill(lclVal),
        lineStyle: { type: 'dashed', color: '#ef4444' },
        symbol: 'none'
    });

    const showSpec = limitConfig && limitConfig.show_spec;
    if (showSpec) {
        if (limitConfig.usl != null) {
            series.push({
                name: 'USL',
                type: 'line',
                data: Array(labels.length).fill(limitConfig.usl),
                lineStyle: { type: 'dotted', color: '#e67e22', width: 1.5 },
                symbol: 'none'
            });
        }
        if (limitConfig.lsl != null) {
            series.push({
                name: 'LSL',
                type: 'line',
                data: Array(labels.length).fill(limitConfig.lsl),
                lineStyle: { type: 'dotted', color: '#e67e22', width: 1.5 },
                symbol: 'none'
            });
        }
    }

    const option = {
        tooltip: { trigger: 'axis' },
        legend: {
            data: series.map(s => s.name),
            left: 'left',
            itemWidth: 15
        },
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
        series: series
    };

    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
}

// ==================== 辅助函数 ====================
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
    const tabs = document.querySelectorAll('.nav-tabs button[data-product]');
    for (let i = 0; i < tabs.length; i++) {
        if (tabs[i].dataset.product === productItem) {
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
    return String(str).replace(/[&<>]/g, m => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;'
    })[m]);
}