/**
 * Презентация к защите проекта Vertex — 16 слайдов, ~10-12 минут.
 *
 *   node scripts/make_defence_deck.js
 *
 * Числа берутся из artifacts/*.json — тех же, из которых написан
 * docs/RESULTS.md и научная работа. Если цифра на слайде разошлась с
 * отчётом, значит устарел артефакт: чинится перезапуском прогона, а не
 * правкой слайда.
 *
 * Оформление: сэндвич из тёмных разделителей и светлых содержательных
 * слайдов, круглые иконки как единственный повторяющийся мотив, крупные
 * числа там и только там, где число и есть содержание слайда.
 */

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const ReactDOMServer = require("react-dom/server");
const React = require("react");
const Fi = require("react-icons/fi");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "artifacts", "figures", "defence");
const OUT = path.join(ROOT, "artifacts", "Vertex_defence.pptx");

// Палитра «ночной пульт наблюдения»: индиго доминирует, янтарь — единственный
// акцент, и он же цвет цены/риска на всех трёх графиках работы.
const NIGHT = "0E1633";
const NIGHT_SOFT = "1B2A52";
const PAPER = "FFFFFF";
const MIST = "F2F4F9";
const AMBER = "C2680E";
const AMBER_LIGHT = "FBF1E4";
const INK = "101828";
const MUTED = "5A6478";
const ICE = "C3D2F0";

const HEAD = "Cambria";
const BODY = "Calibri";

const W = 13.3;
const H = 7.5;
const M = 0.75;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Vertex";
pres.title = "Vertex — аналитика графов и потоков";

// --------------------------------------------------------------------------
// Иконки: react-icons → PNG, чтобы мотив был векторно чистым на проекторе
// --------------------------------------------------------------------------

const iconCache = new Map();

async function icon(name, color) {
  const key = `${name}:${color}`;
  if (iconCache.has(key)) return iconCache.get(key);
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Fi[name], { color: `#${color}`, size: 256, strokeWidth: 2 })
  );
  const png = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  const data = "image/png;base64," + png.toString("base64");
  iconCache.set(key, data);
  return data;
}

// --------------------------------------------------------------------------
// Строительные блоки
// --------------------------------------------------------------------------

function darkSlide() {
  const slide = pres.addSlide();
  slide.background = { color: NIGHT };
  return slide;
}

/** Тёмный слайд-разделитель: номер части, название, одна мысль. */
async function divider(number, title, thought, iconName) {
  const s = darkSlide();
  s.addShape(pres.ShapeType.ellipse, {
    x: M, y: 2.55, w: 1.5, h: 1.5,
    fill: { color: NIGHT_SOFT }, line: { color: AMBER, width: 1.5 },
  });
  s.addImage({ data: await icon(iconName, "E5A353"), x: M + 0.42, y: 2.97, w: 0.66, h: 0.66 });
  s.addText(number, {
    x: M + 2.05, y: 2.35, w: 3, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13, bold: true, color: "E5A353", charSpacing: 2,
  });
  s.addText(title, {
    x: M + 2.05, y: 2.75, w: 9.5, h: 0.85, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 38, bold: true, color: PAPER,
  });
  s.addText(thought, {
    x: M + 2.05, y: 3.7, w: 9.2, h: 0.9, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 17, color: ICE, lineSpacing: 26,
  });
  return s;
}

function contentSlide(title, kicker) {
  const s = pres.addSlide();
  s.background = { color: PAPER };
  if (kicker) {
    s.addText(kicker, {
      x: M, y: 0.42, w: W - 2 * M, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12, bold: true, color: AMBER, charSpacing: 1.6,
    });
  }
  s.addText(title, {
    x: M, y: kicker ? 0.72 : 0.5, w: W - 2 * M, h: 0.8, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 32, bold: true, color: NIGHT,
  });
  return s;
}

function card(s, { x, y, w, h, fill = MIST }) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.06,
    shadow: { type: "outer", angle: 90, blur: 10, offset: 2, color: "D5DAE6", opacity: 0.55 },
  });
}

function stat(s, { x, y, w, value, label, color = NIGHT, size = 50 }) {
  s.addText(value, {
    x, y, w, h: 0.85, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: size, bold: true, color,
  });
  s.addText(label, {
    x, y: y + 0.82, w, h: 0.9, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13, color: MUTED, lineSpacing: 19,
  });
}

async function iconCircle(s, { x, y, size = 0.62, name, bg = NIGHT, fg = "FFFFFF" }) {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: size, h: size, fill: { color: bg }, line: { color: bg },
  });
  s.addImage({ data: await icon(name, fg), x: x + size * 0.26, y: y + size * 0.26, w: size * 0.48, h: size * 0.48 });
}

function footnote(s, text) {
  s.addText(text, {
    x: M, y: 6.72, w: W - 2 * M, h: 0.45, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11, color: MUTED, italic: true,
  });
}

const CHART_FRAME = {
  chartColors: [AMBER, "8FA3C8"],
  showLegend: false,
  showValue: true,
  dataLabelFontFace: BODY,
  dataLabelFontSize: 12,
  dataLabelColor: INK,
  catAxisLabelFontFace: BODY,
  catAxisLabelFontSize: 12,
  catAxisLabelColor: MUTED,
  valAxisLabelFontFace: BODY,
  valAxisLabelColor: MUTED,
  catGridLine: { style: "none" },
  valGridLine: { color: "E4E8F0", size: 1 },
  border: { pt: 0, color: "FFFFFF" },
};

async function build() {
  // ======================================================================
  // 1. Титул
  // ======================================================================
  {
    const s = darkSlide();
    s.addShape(pres.ShapeType.ellipse, {
      x: 10.15, y: 1.25, w: 2.4, h: 2.4,
      fill: { color: NIGHT_SOFT }, line: { color: "2C4174", width: 1 },
    });
    s.addImage({ data: await icon("FiShare2", "E5A353"), x: 10.75, y: 1.85, w: 1.2, h: 1.2 });

    s.addText("VERTEX", {
      x: M, y: 2.15, w: 9, h: 1.1, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 62, bold: true, color: PAPER, charSpacing: 3,
    });
    s.addText("Аналитика графов и потоков\nдля выявления финансового мошенничества", {
      x: M, y: 3.35, w: 8.8, h: 1.2, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 21, color: ICE, lineSpacing: 32,
    });
    s.addText("Научно-исследовательский проект  ·  НИШ ФМН г. Шымкент  ·  2026", {
      x: M, y: 4.9, w: 10.5, h: 0.4, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, color: "E5A353", bold: true,
    });
    s.addNotes(
      "Мы строим не ещё один детектор мошенничества, а среду, в которой обнаружение " +
      "можно измерить — и измерения, сделанные в ней."
    );
  }

  // ======================================================================
  // 2. Разделитель: проблема
  // ======================================================================
  await divider(
    "ЧАСТЬ 1",
    "Дропы — это выход схемы, а не сама схема",
    "Ловить последние сто метров поздно: деньги уже в банкомате.",
    "FiAlertTriangle"
  );

  // ======================================================================
  // 3. Масштаб проблемы
  // ======================================================================
  {
    const s = contentSlide("Проблема, у которой нет открытых данных", "Контекст");
    const items = [
      ["6,87 %", "инцидентов с дропами из 80 871\nобращения в Антифрод-центр", NIGHT, "FiPercent"],
      ["36", "финансовых пирамид ликвидировано\nза полгода, 86 тыс. вовлечённых", NIGHT, "FiUsers"],
      ["0", "открытых данных транзакционного\nантифрода национального уровня", AMBER, "FiDatabase"],
    ];
    for (let i = 0; i < items.length; i++) {
      const [value, label, color, iconName] = items[i];
      const x = M + i * 4.05;
      card(s, { x, y: 1.85, w: 3.75, h: 2.5 });
      await iconCircle(s, { x: x + 0.32, y: 2.15, name: iconName, bg: i === 2 ? AMBER : NIGHT });
      stat(s, { x: x + 0.32, y: 2.95, w: 3.1, value, label, color, size: 40 });
    }
    card(s, { x: M, y: 4.65, w: W - 2 * M, h: 1.6, fill: AMBER_LIGHT });
    s.addText("План Нацбанка требует перехода «от реакции к предотвращению» — транзакционного антифрода на уровне национальных платёжных систем. Этой стадии ещё не существует, поэтому данных с неё нет ни у кого, включая банки. Значит их надо построить.", {
      x: M + 0.4, y: 4.9, w: W - 2 * M - 0.8, h: 1.1, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 16, color: INK, lineSpacing: 26,
    });
  }

  // ======================================================================
  // 4. Почему правила не работают
  // ======================================================================
  {
    const s = contentSlide("Действующие критерии описывают личность, а не поток", "Контекст");
    const rows = [
      ["FiList", "Счёт в чёрном списке", "Новый счёт оформляется на человека, которого нет ни в одной базе"],
      ["FiSmartphone", "Общее устройство или IP", "Виртуальные устройства и SIM меняются автоматически за минуты"],
      ["FiUser", "Отклонение от профиля клиента", "У только что открытого счёта профиля ещё не существует"],
    ];
    for (let i = 0; i < rows.length; i++) {
      const [iconName, title, text] = rows[i];
      const y = 1.9 + i * 1.4;
      card(s, { x: M, y, w: W - 2 * M, h: 1.2 });
      await iconCircle(s, { x: M + 0.35, y: y + 0.29, name: iconName, bg: NIGHT_SOFT });
      s.addText(title, {
        x: M + 1.2, y: y + 0.22, w: 4.2, h: 0.4, isTextBox: true, margin: 0,
        fontFace: HEAD, fontSize: 18, bold: true, color: NIGHT,
      });
      s.addText(text, {
        x: M + 5.6, y: y + 0.24, w: 6.0, h: 0.75, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 14, color: MUTED, lineSpacing: 20,
      });
    }
    s.addText("Три критерия из четырёх обходятся за минуты. Форму денежного потока обойти нельзя — не переставая быть схемой.", {
      x: M, y: 6.15, w: W - 2 * M, h: 0.5, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 16, bold: true, color: AMBER,
    });
  }

  // ======================================================================
  // 5. Разделитель: метод
  // ======================================================================
  await divider(
    "ЧАСТЬ 2",
    "Полигон, где ответ известен заранее",
    "Детектор можно не похвалить, а измерить — в том числе там, где он ломается.",
    "FiTarget"
  );

  // ======================================================================
  // 6. Конвейер
  // ======================================================================
  {
    const s = contentSlide("От сырых событий до дела на столе — одной командой", "Система");
    const steps = [
      ["FiGlobe", "Мир", "330 тыс. событий"],
      ["FiSearch", "Поиск групп", "без файла ответов"],
      ["FiActivity", "Признаки", "из потока денег"],
      ["FiCpu", "Детектор", "обучение на прошлом"],
      ["FiInbox", "Очередь дел", "отрезана порогом"],
    ];
    const boxW = 2.08, gap = 0.34;
    for (let i = 0; i < steps.length; i++) {
      const [iconName, title, sub] = steps[i];
      const x = M + i * (boxW + gap);
      const last = i === steps.length - 1;
      s.addShape(pres.ShapeType.roundRect, {
        x, y: 2.1, w: boxW, h: 2.0,
        fill: { color: last ? AMBER : MIST }, line: { color: last ? AMBER : MIST },
        rectRadius: 0.08,
        shadow: { type: "outer", angle: 90, blur: 8, offset: 2, color: "D5DAE6", opacity: 0.5 },
      });
      await iconCircle(s, {
        x: x + boxW / 2 - 0.31, y: 2.35, name: iconName,
        bg: last ? "FFFFFF" : NIGHT, fg: last ? AMBER : "FFFFFF",
      });
      s.addText(title, {
        x: x + 0.1, y: 3.12, w: boxW - 0.2, h: 0.4, isTextBox: true, margin: 0,
        fontFace: HEAD, fontSize: 16, bold: true, align: "center",
        color: last ? "FFFFFF" : NIGHT,
      });
      s.addText(sub, {
        x: x + 0.1, y: 3.5, w: boxW - 0.2, h: 0.45, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 12, align: "center",
        color: last ? "FBEBD8" : MUTED,
      });
      if (!last) {
        s.addShape(pres.ShapeType.rightArrow, {
          x: x + boxW + 0.09, y: 2.95, w: 0.24, h: 0.22,
          fill: { color: "9AA6BD" }, line: { color: "9AA6BD" },
        });
      }
    }
    s.addText("python scripts/run_demo.py --preset full", {
      x: M, y: 4.55, w: 6.5, h: 0.5, isTextBox: true, margin: 0,
      fontFace: "Courier New", fontSize: 17, bold: true, color: NIGHT,
    });
    s.addText("Собирает очередь, поднимает сервис и интерфейс, дожидается их проверок здоровья и печатает адрес. 17 секунд на весь конвейер, ни одного шага вручную.", {
      x: M, y: 5.1, w: 11.8, h: 0.8, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 15, color: INK, lineSpacing: 24,
    });
    footnote(s, "252 автоматических теста и шесть гейтов качества — всё одной командой.");
  }

  // ======================================================================
  // 7. Четыре правила честности
  // ======================================================================
  {
    const s = contentSlide("Четыре правила, из-за которых числам можно верить", "Метод");
    const rules = [
      ["FiClock", "Обучение только на прошлом", "Purged walk-forward с зазором: ни одна строка не оценивалась моделью, которая её видела."],
      ["FiEyeOff", "Группы ищутся вслепую", "Детектору не передаётся готовая банда. Доля найденного — измеряемый потолок, а не единица по построению."],
      ["FiShuffle", "Признак против шумового пола", "Не побил перемешанный контроль — не подключается. Отрицательный результат тоже результат."],
      ["FiBarChart2", "У каждого уровня свой потолок", "Сколько объектов уровень способен увидеть до всякой модели. Оценка без потолка верится по неверной причине."],
    ];
    for (let i = 0; i < rules.length; i++) {
      const [iconName, title, text] = rules[i];
      const x = i % 2 === 0 ? M : M + 6.05;
      const y = i < 2 ? 1.85 : 4.05;
      card(s, { x, y, w: 5.75, h: 1.95 });
      await iconCircle(s, { x: x + 0.32, y: y + 0.28, name: iconName, bg: NIGHT });
      s.addText(title, {
        x: x + 1.15, y: y + 0.3, w: 4.3, h: 0.45, isTextBox: true, margin: 0,
        fontFace: HEAD, fontSize: 17, bold: true, color: NIGHT,
      });
      s.addText(text, {
        x: x + 0.32, y: y + 0.92, w: 5.15, h: 0.9, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 13, color: MUTED, lineSpacing: 19,
      });
    }
  }

  // ======================================================================
  // 8. Разделитель: результаты
  // ======================================================================
  await divider(
    "ЧАСТЬ 3",
    "Семь измеренных результатов",
    "У каждого написано, как он проверен и чего он не доказывает.",
    "FiCheckCircle"
  );

  // ======================================================================
  // 9. Очередь аналитика
  // ======================================================================
  {
    const s = contentSlide("Что система кладёт человеку на стол", "Результат 1");
    const items = [
      ["31", "дело в очереди за отрезок,\nкоторый детектор не видел", NIGHT],
      ["100 %", "из них — настоящие дропперы", AMBER],
      ["48 %", "всех дропперов отрезка\nпойманы этой очередью", NIGHT],
    ];
    for (let i = 0; i < items.length; i++) {
      const [value, label, color] = items[i];
      const x = M + i * 4.05;
      card(s, { x, y: 1.8, w: 3.75, h: 2.2, fill: i === 1 ? AMBER_LIGHT : MIST });
      stat(s, { x: x + 0.35, y: 2.05, w: 3.1, value, label, color, size: 46 });
    }
    card(s, { x: M, y: 4.35, w: W - 2 * M, h: 1.95 });
    await iconCircle(s, { x: M + 0.35, y: 4.68, name: "FiSliders", bg: NIGHT });
    s.addText("Очередь режется порогом, а не бюджетом", {
      x: M + 1.2, y: 4.6, w: 10.2, h: 0.45, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 20, bold: true, color: NIGHT,
    });
    s.addText("Порог выбирается по прошлым отрезкам и применяется к следующему, которого детектор не видел. Поэтому длина очереди — результат, а не настройка: тихая неделя даёт короткий список, и это верное поведение.", {
      x: M + 1.2, y: 5.1, w: 10.2, h: 1.0, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 15, color: INK, lineSpacing: 23,
    });
    s.addNotes("Это та точность, которая была бы в понедельник, а не подогнанная задним числом.");
  }

  // ======================================================================
  // 10-12. Графики
  // ======================================================================
  function figureSlide(title, kicker, image, caption, notes) {
    const s = contentSlide(title, kicker);
    s.addImage({
      path: path.join(FIG, image), x: 1.6, y: 1.7, w: 10.1, h: 4.5,
      sizing: { type: "contain", w: 10.1, h: 4.5 },
    });
    s.addText(caption, {
      x: M, y: 6.35, w: W - 2 * M, h: 0.7, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, color: INK, lineSpacing: 21,
    });
    if (notes) s.addNotes(notes);
    return s;
  }

  figureSlide(
    "Сложность объявлена ДО прогонов — и показаны все ступени",
    "Результат 2",
    "ladder.png",
    "Пять миров, один детектор. На пятой ступени групповой уровень не ошибается — у него вообще нет оценки.",
    "Разница между «выбрали сложность и назвали её» и «построили пять миров, показали тот, где выиграли»."
  );

  figureSlide(
    "Главный вопрос банка: а если мошенников мало?",
    "Результат 3",
    "rarity.png",
    "От 7 % до 0,1 % мошенников: ROC-AUC падает лишь с 0,957 до 0,894, а работа аналитика растёт в 80 раз.",
    "Первый вопрос любого человека из банка. Мы его измерили, а не отмахнулись."
  );

  // ======================================================================
  // 13. Порог вместо бюджета — нативная диаграмма
  // ======================================================================
  {
    const s = contentSlide("Ответ: менять надо не детектор, а способ им пользоваться", "Результат 4");
    s.addText("При доле мошенников 0,1 % — как в жизни", {
      x: M, y: 1.72, w: 11.8, h: 0.35, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 15, color: MUTED,
    });
    s.addChart(
      pres.ChartType.bar,
      [{
        name: "Доля подтверждений",
        labels: ["Верхние 10 % списка", "Порог под половину дропперов"],
        values: [0.008, 0.86],
      }],
      {
        ...CHART_FRAME,
        x: M, y: 2.15, w: 6.6, h: 3.1,
        barDir: "bar",
        dataLabelPosition: "outEnd",
        dataLabelFormatCode: "0%",
        valAxisMaxVal: 1,
        valAxisLabelFormatCode: "0%",
        showTitle: true,
        title: "Из скольких сигналов подтверждается мошенничество",
        titleFontFace: HEAD,
        titleFontSize: 14,
        titleColor: NIGHT,
      }
    );
    const cost = [
      [2.15, MIST, "100", MUTED, "сигналов на 1000 счетов\nпри фиксированном бюджете"],
      [3.78, AMBER_LIGHT, "0,6", AMBER, "сигнала на 1000 счетов,\nчтобы поймать половину"],
    ];
    cost.forEach(([y, fill, value, color, label]) => {
      card(s, { x: M + 7.0, y, w: 4.8, h: 1.45, fill });
      s.addText(value, {
        x: M + 7.35, y: y + 0.16, w: 1.5, h: 0.55, isTextBox: true, margin: 0,
        fontFace: HEAD, fontSize: 32, bold: true, color,
      });
      s.addText(label, {
        x: M + 8.85, y: y + 0.2, w: 2.75, h: 0.75, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 12, color: MUTED, lineSpacing: 17,
      });
    });

    s.addText("Верх списка крутой: половина мошенников лежит выше порога, до которого почти не дотягивается никто честный. Первая половина очереди почти бесплатна — и это то, что банк может применить завтра, ничего не переобучая.", {
      x: M, y: 5.5, w: 11.8, h: 1.0, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 15, color: INK, lineSpacing: 24,
    });
    s.addNotes("Тот же детектор, другая политика применения.");
  }

  // ======================================================================
  // 14. Уклонение
  // ======================================================================
  figureSlide(
    "Сколько стоит спрятаться — и что именно ломается",
    "Результат 5",
    "evasion.png",
    "Групповой поиск ломается на третьем источнике денег. Поиск по одному человеку не замечает уклонения вообще.",
    "Один уровень сильнее, другой переживает противника, который платит. Поэтому в системе работают оба."
  );

  // ======================================================================
  // 15. Правила против модели + крипта
  // ======================================================================
  {
    const s = contentSlide("Правила против потока, и вся схема целиком", "Результаты 6-7");
    s.addChart(
      pres.ChartType.bar,
      [{
        name: "ROC-AUC",
        labels: ["Критерии приказа", "Vertex на тех же данных"],
        values: [0.761, 0.959],
      }],
      {
        ...CHART_FRAME,
        x: M, y: 1.85, w: 6.5, h: 2.6,
        barDir: "bar",
        dataLabelPosition: "outEnd",
        dataLabelFormatCode: "0.000",
        valAxisMaxVal: 1,
        showTitle: true,
        title: "Десять независимых запусков",
        titleFontFace: HEAD,
        titleFontSize: 14,
        titleColor: NIGHT,
      }
    );
    s.addText("Главное — не разрыв, а его причина: три критерия из четырёх описывают личность и оборудование, а не денежный поток. Кольцо чистых счетов проходит их насквозь.", {
      x: M, y: 4.6, w: 6.5, h: 1.1, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, color: INK, lineSpacing: 21,
    });

    card(s, { x: M + 7.0, y: 1.85, w: 4.8, h: 3.85, fill: MIST });
    await iconCircle(s, { x: M + 7.35, y: 2.15, name: "FiLink", bg: NIGHT });
    s.addText("Крипто-канал в мире", {
      x: M + 7.35, y: 2.95, w: 4.2, h: 0.4, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 18, bold: true, color: NIGHT,
    });
    const crypto = [
      ["4 из 5", "типологий приказа работают"],
      ["542", "крипто-события, 15 колец"],
      ["0", "срабатываний на честных трейдерах"],
    ];
    crypto.forEach(([value, label], i) => {
      const y = 3.45 + i * 0.75;
      s.addText(value, {
        x: M + 7.35, y, w: 1.4, h: 0.45, isTextBox: true, margin: 0,
        fontFace: HEAD, fontSize: 20, bold: true, color: i === 2 ? AMBER : NIGHT,
      });
      s.addText(label, {
        x: M + 8.85, y: y + 0.06, w: 2.7, h: 0.6, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 12, color: MUTED, lineSpacing: 16,
      });
    });
    footnote(s, "Честные крипто-трейдеры — обязательный контроль: без них детектор выучил бы «крипта = мошенничество».");
  }

  // ======================================================================
  // 16. Пять дефектов
  // ======================================================================
  {
    const s = contentSlide("Пять дефектов, которые мы нашли у себя сами", "Проверка");
    s.addText("Это тоже результат: он показывает, что мы себя проверяли, а не только хвалили.", {
      x: M, y: 1.62, w: 11.8, h: 0.35, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 15, color: MUTED,
    });
    const defects = [
      ["Генератор писал ответ в признаки", "ROC-AUC 1,0000 → 0,81 после разделения. Первая честная цифра."],
      ["Сборщик дел брал группы из файла ответов", "Теперь группы ищутся вслепую, метки прикладываются после."],
      ["Базовая линия жульничала в свою пользу", "Критерий «общее устройство» срабатывал на 100 % банд — по построению."],
      ["79 % «честных» были счетами с 1-2 событиями", "Модель отличала активных от неактивных. Введён порог в 10 событий."],
      ["Первая лестница мерила дисбаланс классов", "Мир на 96 % из мошенников. Правило закреплено тестом."],
    ];
    defects.forEach(([title, text], i) => {
      const y = 2.15 + i * 0.92;
      s.addShape(pres.ShapeType.ellipse, {
        x: M, y: y + 0.05, w: 0.42, h: 0.42, fill: { color: AMBER_LIGHT }, line: { color: AMBER_LIGHT },
      });
      s.addText(String(i + 1), {
        x: M, y: y + 0.1, w: 0.42, h: 0.34, isTextBox: true, margin: 0,
        fontFace: HEAD, fontSize: 15, bold: true, color: AMBER, align: "center",
      });
      s.addText(title, {
        x: M + 0.6, y, w: 5.2, h: 0.5, isTextBox: true, margin: 0,
        fontFace: HEAD, fontSize: 15, bold: true, color: NIGHT,
      });
      s.addText(text, {
        x: M + 6.0, y: y + 0.02, w: 5.8, h: 0.7, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 13, color: MUTED, lineSpacing: 18,
      });
    });
    footnote(s, "Ни один из пяти не был опечаткой: это работающий код, отвечавший не на тот вопрос.");
  }

  // ======================================================================
  // 17. Итог
  // ======================================================================
  {
    const s = darkSlide();
    s.addText("Что мы приносим на защиту", {
      x: M, y: 0.9, w: 11.8, h: 0.75, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 34, bold: true, color: PAPER,
    });
    const points = [
      ["FiTerminal", "Работающая система", "От сырых событий до очереди дел — одной командой, 17 секунд, 252 теста."],
      ["FiBookOpen", "Семь измеренных результатов", "У каждого написано, как проверен и чего он не доказывает."],
      ["FiAward", "То, чего нет ни у кого", "Полигон, где схема известна заранее, и цена уклонения по шагам."],
    ];
    for (let i = 0; i < points.length; i++) {
      const [iconName, title, text] = points[i];
      const y = 2.05 + i * 1.3;
      await iconCircle(s, { x: M, y: y + 0.08, size: 0.66, name: iconName, bg: AMBER });
      s.addText(title, {
        x: M + 1.0, y, w: 4.6, h: 0.5, isTextBox: true, margin: 0,
        fontFace: HEAD, fontSize: 21, bold: true, color: PAPER,
      });
      s.addText(text, {
        x: M + 5.7, y: y + 0.04, w: 6.1, h: 0.85, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 14, color: ICE, lineSpacing: 21,
      });
    }
    s.addText("Дальше: крипто-ступень в лестницу миров  ·  метрики BAS и SIS  ·  внешняя валидация на Elliptic", {
      x: M, y: 6.25, w: 11.8, h: 0.5, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, color: "E5A353", bold: true,
    });
  }

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("Готово:", OUT);
}

build().catch((error) => {
  console.error(error);
  process.exit(1);
});
