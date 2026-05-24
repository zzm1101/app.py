// static/js/spc.js
// SPC分析前端脚本（颜色统一为紫色系）

$(function () {
    var xbarChart, rChart, histChart, qqChart, trendChart;
    var $ctqSelect = $('#ctqSelect');
    var $productItemSelect = $('#productItemSelect');
    var $productItemHint = $('#productItemHint');
    var $loadBtn = $('#loadBtn');
    var $resetRules = $('#resetRules');
    var $uslInput = $('#uslInput');
    var $lslInput = $('#lslInput');
    var $targetInput = $('#targetInput');
    var $resetSpecBtn = $('#resetSpecBtn');
    var $chartType = $('#chartType');
    var $dataWarning = $('#dataWarning');
    var $startDate = $('#startDate');
    var $endDate = $('#endDate');
    var $trendGranularity = $('#trendGranularity');
    var $useBoxCox = $('#useBoxCox');
    var currentMetric = 'ppk';

    // 全选按钮
    $resetRules.click(function () { $('.rule-check').prop('checked', true); });

    // CTQ切换
    $ctqSelect.change(function () {
        var ctqId = $(this).val();
        var selectedOption = $(this).find('option:selected');
        var ctqProductItem = selectedOption.data('product-item') || '';
        if (ctqId) {
            $.get('/ctq/api/' + ctqId, function (ctq) {
                $uslInput.val(ctq.usl);
                $lslInput.val(ctq.lsl);
                $targetInput.val(ctq.target_m);
            }).fail(function() { alert('获取CTQ信息失败'); });
            if (ctqProductItem && ctqProductItem !== '通用') {
                $productItemSelect.val(ctqProductItem).prop('disabled', true);
                $productItemHint.text('此CTQ已绑定品项：' + ctqProductItem + '（自动锁定）');
            } else {
                $productItemSelect.prop('disabled', false).val('');
                $productItemHint.text('通用CTQ，请务必选择一个品项');
            }
        } else {
            $productItemSelect.prop('disabled', false).val('');
            $productItemHint.text('');
        }
    });

    $resetSpecBtn.click(function () {
        var ctqId = $ctqSelect.val();
        if (!ctqId) return;
        $.get('/ctq/api/' + ctqId, function (ctq) {
            $uslInput.val(ctq.usl);
            $lslInput.val(ctq.lsl);
            $targetInput.val(ctq.target_m);
        });
    });

    // 主分析按钮
    $loadBtn.click(function () {
        var ctqId = $ctqSelect.val();
        if (!ctqId) { alert('请选择CTQ'); return; }
        var selectedProductItem = $productItemSelect.val();
        if (!selectedProductItem) { alert('请选择一个品项'); return; }
        var params = {
            ctq_id: ctqId,
            product_item: selectedProductItem,
            chart_type: $chartType.val(),
            usl: $uslInput.val() || undefined,
            lsl: $lslInput.val() || undefined,
            target: $targetInput.val() || undefined,
            rules: $('.rule-check:checked').map(function(){return this.value;}).get().join(','),
            start_date: $startDate.val(),
            end_date: $endDate.val(),
            use_boxcox: $useBoxCox.is(':checked') ? 1 : 0
        };
        $loadBtn.prop('disabled', true).html('<i class="fa fa-spinner fa-spin"></i> 加载中...');
        $dataWarning.hide();
        $.ajax({
            url: '/spc/data',
            data: params,
            timeout: 60000,
            dataType: 'json',
            success: function(res) {
                if (res.error) { alert(res.error); return; }
                if (res.message) $dataWarning.removeClass('d-none').html('<i class="fa fa-info-circle"></i> '+res.message);
                else $dataWarning.addClass('d-none');
                if (res.warning) $dataWarning.removeClass('d-none').append('<br><i class="fa fa-exclamation-triangle"></i> '+res.warning);
                try { drawAllCharts(res); } catch(e) { console.error(e); }
                try { updateCapabilityCards(res); } catch(e) { console.error(e); }
                try { renderCapabilityTable(res); } catch(e) { console.error(e); }
                try { updateAlarmTable(res); } catch(e) { console.error(e); }
                try { updateNormality(res); } catch(e) { console.error(e); }
                // 如果趋势tab激活，自动刷新
                if ($('#tabTrend').hasClass('active')) loadCapabilityTrend();
            },
            error: function(xhr, status, err) {
                var msg = status === 'timeout' ? '分析超时，请缩小时间范围' : (xhr.responseJSON ? xhr.responseJSON.error : err);
                alert('加载失败：' + msg);
            },
            complete: function() {
                $loadBtn.prop('disabled', false).html('<i class="fa fa-play"></i> 分析');
            }
        });
    });

    // Box-Cox变化时，如果趋势tab激活则自动刷新趋势
    $useBoxCox.change(function() {
        if ($('#tabTrend').hasClass('active')) loadCapabilityTrend();
    });

    // ---------- 绘图函数 ----------
    function drawAllCharts(data) {
        if (xbarChart && !xbarChart.isDisposed()) xbarChart.dispose();
        if (rChart && !rChart.isDisposed()) rChart.dispose();
        if (histChart && !histChart.isDisposed()) histChart.dispose();

        var dates = data.dates.length ? data.dates : data.labels;
        var n = dates.length;

        // 均值控制图
        xbarChart = echarts.init(document.getElementById('xbarChart'));
        var xbarSeries = [
            { name: '均值', type: 'line', data: data.xbar, symbol: 'circle', symbolSize: 6, itemStyle: {color:'#6366f1'},
              markPoint: { data: (data.alarm_x||[]).map(i=>({name:'异常',coord:[i,data.xbar[i]],symbol:'circle',symbolSize:12,itemStyle:{color:'#ef4444'}})) } },
            { name: 'UCL', type: 'line', data: Array(n).fill(data.ucl_x), lineStyle:{type:'dashed',color:'#ef4444',width:2}, symbol:'none' },
            { name: 'LCL', type: 'line', data: Array(n).fill(data.lcl_x), lineStyle:{type:'dashed',color:'#ef4444',width:2}, symbol:'none' },
            { name: 'CL', type: 'line', data: Array(n).fill(data.cl_x), lineStyle:{type:'dashed',color:'#10b981',width:1.5}, symbol:'none' }
        ];
        if (data.usl) { xbarSeries.push({name:'USL',type:'line',data:Array(n).fill(data.usl),lineStyle:{type:'solid',color:'#10b981',width:1.5},symbol:'none'}); }
        if (data.lsl) { xbarSeries.push({name:'LSL',type:'line',data:Array(n).fill(data.lsl),lineStyle:{type:'solid',color:'#10b981',width:1.5},symbol:'none'}); }
        if (data.target) { xbarSeries.push({name:'目标',type:'line',data:Array(n).fill(data.target),lineStyle:{type:'dashed',color:'#6366f1',width:1.5},symbol:'none'}); }
        xbarChart.setOption({
            title: { text: data.ctq_name + ' ' + data.title_xbar + ' (' + data.chart_type + ')', left:'center', textStyle:{fontSize:12,color:'#334155'} },
            tooltip: { trigger:'axis' },
            legend: { data: ['均值','UCL','LCL','CL','USL','LSL','目标'].filter(d=>d), bottom:0 },
            grid: { left:'8%', right:'5%', top:'18%', bottom:'12%' },
            xAxis: { type:'category', data: dates },
            yAxis: { type:'value', scale:true },
            series: xbarSeries
        });

        // 极差/标准差图
        var rVals = data.r.map(v => v !== null ? v : 0);
        rChart = echarts.init(document.getElementById('rChart'));
        rChart.setOption({
            title: { text: data.title_r + ' (' + data.chart_type + ')', left:'center', textStyle:{fontSize:12,color:'#334155'} },
            tooltip: { trigger:'axis' },
            legend: { data: [data.r_label,'UCL','CL'], bottom:0 },
            grid: { left:'8%', right:'5%', top:'18%', bottom:'12%' },
            xAxis: { type:'category', data: dates },
            yAxis: { type:'value', scale:true },
            series: [
                { name: data.r_label, type:'line', data: rVals, symbol:'circle', symbolSize:6, itemStyle:{color:'#f59e0b'} },
                { name: 'UCL', type:'line', data: Array(n).fill(data.ucl_r), lineStyle:{type:'dashed',color:'#ef4444',width:2}, symbol:'none' },
                { name: 'CL', type:'line', data: Array(n).fill(data.cl_r), lineStyle:{type:'dashed',color:'#10b981',width:1.5}, symbol:'none' }
            ]
        });

        // 直方图 - 修复 OOM：限制正态拟合最多 500 个点
        histChart = echarts.init(document.getElementById('histogram'));
        var vals = data.all_values.slice().sort((a,b)=>a-b);
        var m = data.mean, s = data.std_overall || 0.0001;
        var bins = Math.ceil(Math.sqrt(vals.length)) + 1;
        var minV = vals[0], maxV = vals[vals.length-1];
        var step = (maxV - minV) / bins;
        var histData = [];
        for (var i=0; i<bins; i++) {
            var low = minV + i*step, high = low + step;
            histData.push([(low+high)/2, vals.filter(v => v >= low && v < high).length]);
        }
        // 正态拟合曲线：动态调整步长，确保最多 500 个点
        var normalData = [];
        if (s > 0) {
            var startX = m - 4*s, endX = m + 4*s;
            var desiredPoints = 500;
            var dynamicStep = (endX - startX) / desiredPoints;
            if (dynamicStep < 0.001) dynamicStep = 0.001;
            for (var x = startX; x <= endX; x += dynamicStep) {
                normalData.push([x, normalDensity(x, m, s) * vals.length * step]);
            }
        }
        histChart.setOption({
            tooltip: { trigger:'axis' },
            legend: { data: ['频数','正态拟合','USL','LSL','目标'], top:5 },
            grid: { left:'10%', right:'5%', top:'15%', bottom:'10%' },
            xAxis: { type:'value', scale:true },
            yAxis: { type:'value', name:'频数' },
            series: [
                { name:'频数', type:'bar', data: histData, barWidth:'90%', itemStyle:{color:'#c7d2fe'} },
                { name:'正态拟合', type:'line', data: normalData, smooth:true, lineStyle:{color:'#10b981'}, symbol:'none' },
                { name:'USL', type:'line', markLine:{silent:true,symbol:'none',lineStyle:{type:'dashed',color:'#ef4444',width:2},data:[{xAxis:data.usl}]} },
                { name:'LSL', type:'line', markLine:{silent:true,symbol:'none',lineStyle:{type:'dashed',color:'#ef4444',width:2},data:[{xAxis:data.lsl}]} },
                { name:'目标', type:'line', markLine:{silent:true,symbol:'none',lineStyle:{type:'dashed',color:'#6366f1',width:2},data:[{xAxis:data.target}]} }
            ]
        });

        window.addEventListener('resize', function() {
            if (xbarChart && !xbarChart.isDisposed()) xbarChart.resize();
            if (rChart && !rChart.isDisposed()) rChart.resize();
            if (histChart && !histChart.isDisposed()) histChart.resize();
        });
    }

    function normalDensity(x, mean, std) {
        return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
    }

    function updateCapabilityCards(data) {
        var setCap = function(id, value, thresholds) {
            var el = $('#' + id);
            el.text(value != null ? value.toFixed(4) : '-');
            var capCard = el.closest('.capability-card');
            var color = '#333', bg = '#f8fafc';
            if (value != null && thresholds) {
                if (value >= thresholds.good) { color = '#059669'; bg = '#ecfdf5'; }
                else if (value >= thresholds.warn) { color = '#d97706'; bg = '#fffbeb'; }
                else { color = '#dc2626'; bg = '#fef2f2'; }
            }
            capCard.css({ 'background-color': bg, transition: 'background-color 0.2s' });
            el.css('color', color);
        };
        setCap('cpkValue', data.cpk, { good: 1.33, warn: 1.0 });
        setCap('ppkValue', data.ppk, { good: 1.33, warn: 1.0 });
        setCap('cpmValue', data.cpm, { good: 1.33, warn: 1.0 });
        $('#meanValue').text(data.mean.toFixed(4));
        $('#capabilityCards').removeClass('d-none');
    }

    function renderCapabilityTable(data) {
    var html = '<table class="table table-sm table-bordered small"><thead><tr><th colspan="2">过程能力分析</th></tr></thead><tbody>';
    html += '<tr><td>原始样本数</td><td>'+data.all_values.length+'</td></tr>';
    html += '<tr><td>监测批次数</td><td>'+data.labels.length+'</td></tr>';
    html += '<tr><td>均值</td><td>'+data.mean.toFixed(4)+'</td></tr>';
    html += '<tr><td>标准差(组内)</td><td>'+(data.std_within?data.std_within.toFixed(4):'-')+'</td></tr>';
    html += '<tr><td>标准差(整体)</td><td>'+data.std_overall.toFixed(6)+'</td></tr>';
    html += '<tr><td>USL</td><td>'+(data.usl!==undefined?data.usl.toFixed(4):'-')+'</td></tr>';
    html += '<tr><td>LSL</td><td>'+(data.lsl!==undefined?data.lsl.toFixed(4):'-')+'</td></tr>';
    html += '<tr><td>目标</td><td>'+(data.target!==undefined?data.target.toFixed(4):'-')+'</td></tr>';
    html += '<tr><td>Cp</td><td>'+(data.cp!=null?data.cp.toFixed(4):'-')+'</td></tr>';
    // 修复：增加 "CPK" 标签列
    html += '<tr><td>CPK</td><td class="fw-bold '+(data.cpk>=1.33?'text-success':'text-danger')+'">'+(data.cpk!=null?data.cpk.toFixed(4):'-')+'</td></tr>';
    html += '<tr><td>Pp</td><td>'+(data.pp!=null?data.pp.toFixed(4):'-')+'</td></tr>';
    html += '<tr><td>Ppk</td><td>'+(data.ppk!=null?data.ppk.toFixed(4):'-')+'</td></tr>';
    html += '<tr><td>Cpm</td><td>'+(data.cpm!=null?data.cpm.toFixed(4):'-')+'</td></tr>';
    html += '<tr><td>Cpmk</td><td>'+(data.cpmk!=null?data.cpmk.toFixed(4):'-')+'</td></tr>';
    html += '<tr><td>Z.Bench</td><td>'+(data.z_bench!=null?data.z_bench.toFixed(2):'-')+'</td></tr>';
    html += '<tr><td>PPM 总计</td><td>'+(data.ppm_total!=null?data.ppm_total.toFixed(0):'-')+'</td></tr>';
    html += '<tr><td>PPM > USL</td><td>'+(data.ppm_usl!=null?data.ppm_usl.toFixed(0):'-')+'</td></tr>';
    html += '<tr><td>PPM < LSL</td><td>'+(data.ppm_lsl!=null?data.ppm_lsl.toFixed(0):'-')+'</td></tr>';
    html += '</tbody></table>';
    $('#capabilityStatsTable').html(html);
}
    function updateAlarmTable(data) {
        var container = $('#alarmTableContainer');
        if (!data.rules_violations || data.rules_violations.length === 0) {
            container.html('<div class="alert alert-success py-1">未检测到异常点</div>');
            return;
        }
        var html = '<table class="table table-sm table-striped"><thead><tr><th>批次/子组</th><th>数值</th><th>违反规则</th></tr></thead><tbody>';
        data.rules_violations.forEach(function(v) {
            var parts = v.split(' 违反规则');
            var left = parts[0].replace('批次','').replace(' 均值',' ');
            var batch = left.split(' ')[0];
            var val = left.split(' ')[1] || '';
            var rules = parts.length > 1 ? parts[1] : '';
            html += '<tr><td>'+batch+'</td><td>'+val+'</td><td>'+rules+'</td></tr>';
        });
        html += '</tbody></table>';
        container.html(html);
    }

    function updateNormality(data) {
        $('#swStat').text(data.sw_stat || '-');
        $('#swP').text(data.sw_p || '-');
        $('#adStat').text(data.ad_stat || '-');
        $('#adCrit').text(data.ad_crit_5 || '-');
        var passed = (data.sw_p && data.sw_p > 0.05) && data.ad_pass;
        $('#normConclusion').text(passed ? '数据符合正态分布' : '数据可能偏离正态分布');

        if (qqChart && !qqChart.isDisposed()) qqChart.dispose();
        if (data.qq_points && data.qq_points.length > 0) {
            qqChart = echarts.init(document.getElementById('qqChart'));
            var theory = data.qq_points.map(p => p[0]);
            var sample = data.qq_points.map(p => p[1]);
            var n = theory.length;
            var sumX = theory.reduce((a,b)=>a+b,0), sumY = sample.reduce((a,b)=>a+b,0);
            var sumXY = theory.reduce((a,b,i)=>a+b*sample[i],0), sumX2 = theory.reduce((a,b)=>a+b*b,0);
            var slope = (n*sumXY - sumX*sumY)/(n*sumX2 - sumX*sumX);
            var intercept = (sumY - slope*sumX)/n;
            var minX = Math.min(...theory), maxX = Math.max(...theory);
            qqChart.setOption({
                title: { text: 'Q‑Q Plot', left:'center', textStyle:{color:'#334155'} },
                tooltip: { trigger:'axis' },
                xAxis: { name:'理论分位数' },
                yAxis: { name:'样本分位数' },
                series: [
                    { name:'数据点', type:'scatter', data:data.qq_points, symbolSize:6, itemStyle:{color:'#6366f1'} },
                    { name:'参考线', type:'line', data:[[minX,slope*minX+intercept],[maxX,slope*maxX+intercept]], lineStyle:{color:'#ef4444',type:'dashed'}, symbol:'none' }
                ]
            });
        }
    }

    // ---------- 能力趋势（健壮版） ----------
    function loadCapabilityTrend() {
        var ctqId = $ctqSelect.val();
        var productItem = $productItemSelect.val();
        if (!ctqId || !productItem) {
            $('#trendChart').html('<div class="alert alert-warning">请先选择CTQ和品项</div>');
            return;
        }
        var startDate = $startDate.val();
        var endDate = $endDate.val();
        var granularity = $trendGranularity.val();
        var useBoxCox = $useBoxCox.is(':checked') ? 1 : 0;

        $('#trendChart').html('<div class="text-center"><i class="fa fa-spinner fa-spin"></i> 加载中...</div>');

        $.ajax({
            url: '/spc/capability_trend',
            data: {
                ctq_id: ctqId,
                product_item: productItem,
                start_date: startDate,
                end_date: endDate,
                granularity: granularity,
                use_boxcox: useBoxCox
            },
            success: function(res) {
                if (res.error) {
                    $('#trendChart').html('<div class="alert alert-warning">' + res.error + '</div>');
                    return;
                }
                if (!res.ppk && !res.cpm) {
                    $('#trendChart').html('<div class="alert alert-warning">返回数据格式无效</div>');
                    return;
                }
                drawTrendChart(res);
            },
            error: function(xhr) {
                var msg = '加载趋势数据失败';
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    msg = xhr.responseJSON.error;
                } else if (xhr.status === 400) {
                    msg = xhr.responseText;
                }
                $('#trendChart').html('<div class="alert alert-danger">' + msg + '</div>');
            }
        });
    }

    function drawTrendChart(data) {
        var container = document.getElementById('trendChart');
        if (!container) {
            console.error('trendChart 容器不存在');
            $('#trendChart').html('<div class="alert alert-danger">图表容器未找到，请刷新页面重试</div>');
            return;
        }

        // 安全销毁旧图表
        if (trendChart) {
            try {
                if (typeof trendChart.dispose === 'function') {
                    trendChart.dispose();
                }
            } catch (e) {
                console.warn('销毁旧图表失败', e);
            }
            trendChart = null;
        }

        var metric = currentMetric;
        var yDataRaw = data[metric];
        var labels = data.labels || [];
        var targetCpk = data.target_cpk || 1.33;

        // 将 null 和 NaN 转换为 undefined（ECharts 不会绘制该点）
        var yData = yDataRaw.map(function(v) {
            if (v === null || v === undefined || isNaN(v)) return undefined;
            return v;
        });

        var hasValid = yData.some(function(v) { return v !== undefined; });
        if (!hasValid || labels.length === 0) {
            $('#trendChart').html('<div class="alert alert-warning">当前指标无有效数据，请切换其他指标或检查数据（每个分组至少需要2个样本）</div>');
            return;
        }

        try {
            trendChart = echarts.init(container);
            var targetLine = Array(yData.length).fill(targetCpk);
            trendChart.setOption({
                tooltip: {
                    trigger: 'axis',
                    formatter: function(params) {
                        if (!params || params.length === 0) return '';
                        var val = params[0].value;
                        if (val === undefined) return params[0].axisValue + '<br/>无数据';
                        return params[0].axisValue + '<br/>' + metric.toUpperCase() + ': ' + val.toFixed(4);
                    }
                },
                xAxis: { type: 'category', data: labels, axisLabel: { rotate: 30, interval: 0, fontSize: 10 } },
                yAxis: { type: 'value', name: metric.toUpperCase(), min: 0 },
                series: [
                    {
                        name: metric.toUpperCase(),
                        type: 'line',
                        data: yData,
                        connectNulls: false,
                        lineStyle: { width: 2, color: '#6366f1' },
                        itemStyle: { color: '#6366f1' },
                        markPoint: { data: [{ type: 'min', name: '最小值' }] }
                    },
                    {
                        name: '目标值',
                        type: 'line',
                        data: targetLine,
                        lineStyle: { type: 'dashed', color: '#ef4444' },
                        symbol: 'none'
                    }
                ]
            });
        } catch (e) {
            console.error('图表初始化失败', e);
            $('#trendChart').html('<div class="alert alert-danger">图表渲染失败：' + e.message + '</div>');
        }
    }

    // 选项卡切换
    $('#chartTabs a[data-bs-toggle="tab"]').on('shown.bs.tab', function(e) {
        var target = $(e.target).attr('href');
        if (target === '#tabControl') { if (xbarChart && !xbarChart.isDisposed()) xbarChart.resize(); if (rChart && !rChart.isDisposed()) rChart.resize(); }
        else if (target === '#tabHist') { if (histChart && !histChart.isDisposed()) histChart.resize(); }
        else if (target === '#tabNormality') { if (qqChart && !qqChart.isDisposed()) qqChart.resize(); }
        else if (target === '#tabTrend') { loadCapabilityTrend(); }
    });

    // 趋势指标切换
    $(document).on('click', '[data-metric]', function() {
        $('[data-metric]').removeClass('active');
        $(this).addClass('active');
        currentMetric = $(this).data('metric');
        loadCapabilityTrend();
    });

    $('#refreshTrendBtn').click(function() { loadCapabilityTrend(); });

    // 历史记录
    $('#btnHistory').click(function() {
        $.getJSON('/spc/history', function(res) {
            var html = '<table class="table table-sm"><thead><tr><th>时间</th><th>CTQ</th><th>类型</th><th>范围</th><th>操作</th></tr></thead><tbody>';
            res.data.forEach(function(r) {
                html += '<tr><td>'+r.analysis_time+'</td><td>'+r.ctq_name+'</td><td>'+r.chart_type+'</td><td>'+r.time_range+'</td><td><button class="btn btn-sm btn-outline-primary load-history" data-id="'+r.id+'">加载</button></td></tr>';
            });
            html += '</tbody></table>';
            $('#historyTableBody').html(html);
            $('#historyModal').modal('show');
        });
    });

    $(document).on('click', '.load-history', function() {
        var id = $(this).data('id');
        $.getJSON('/spc/history/' + id, function(res) {
            try { drawAllCharts(res); } catch(e) {}
            try { updateCapabilityCards(res); } catch(e) {}
            try { renderCapabilityTable(res); } catch(e) {}
            try { updateAlarmTable(res); } catch(e) {}
            try { updateNormality(res); } catch(e) {}
            $('#historyModal').modal('hide');
        });
    });
});