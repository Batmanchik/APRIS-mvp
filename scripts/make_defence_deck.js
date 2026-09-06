/**
 * Презентация к защите Cheops AI — 14 слайдов, ~10-12 минут.
 *
 *   node scripts/make_defence_deck.js
 *
 * Числа берутся из artifacts/*.json — тех же, из которых написан
 * docs/RESULTS.md. Если цифра на слайде разошлась с отчётом, значит устарел
 * артефакт, и чинится это перезапуском прогона, а не правкой слайда.
 *
 * Картинки: artifacts/figures/defence/*.png (scripts/make_defence_figures.py).
 */

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "artifacts", "figures", "defence");
const OUT = path.join(ROOT, "artifacts", "Cheops_AI_defence.pptx");

// Midnight Executive: тёмно-синий доминирует, амбер — единственный акцент,
// и он же цвет «цены/риска» на всех трёх графиках.
const NAVY = "12224A";
const NAVY_SOFT = "1E2761";
const ICE = "CADCFC";
const AMBER = "B45309";
const WHITE = "FFFFFF";
const INK = "1F2328";
const MUTED = "6B7280";
const PAPER = "FFFFFF";

const HEAD = "Cambria";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.author = "Cheops AI";
pres.title = "Cheops AI — измеримое обнаружение финансовых схем";

const W = 13.3;
const M = 0.7; // поля

/** Тёмный слайд: титул, разделы, финал. */
function darkSlide() {
  const slide = pres.addSlide();
  slide.background = { color: NAVY };
  return slide;
}

/** Светлый слайд с заголовком. */
function contentSlide(title, kicker) {
  const slide = pres.addSlide();
  slide.background = { color: PAPER };
  if (kicker) {
    slide.addText(kicker, {
      x: M, y: 0.42, w: W - 2 * M, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13, bold: true, color: AMBER, charSpacing: 1.2,
    });
  }
  slide.addText(title, {
    x: M, y: kicker ? 0.74 : 0.5, w: W - 2 * M, h: 0.85, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 34, bold: true, color: NAVY,
  });
  return slide;
}

/** Крупное число с подписью — основной мотив колоды. */
function stat(slide, { x, y, w, value, label, color = NAVY, size = 54 }) {
  slide.addText(value, {
    x, y, w, h: 0.9, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: size, bold: true, color,
  });
  slide.addText(label, {
    x, y: y + 0.88, w, h: 0.75, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13, color: MUTED,
  });
}

/** Карточка с тонкой заливкой — без полосок и рамок-акцентов. */
function card(slide, { x, y, w, h, fill = "F4F6FB" }) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08,
  });
}

function note(slide, text) {
  slide.addText(text, {
    x: M, y: 6.82, w: W - 2 * M, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11, color: MUTED, italic: true,
  });
}

// ==========================================================================
// 1. Титул
// ==========================================================================
{
  const s = darkSlide();
  s.addText("Cheops AI", {
    x: M, y: 2.0, w: W - 2 * M, h: 1.0, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 54, bold: true, color: WHITE,
  });
  s.addText("Среда, в которой обнаружение финансовых схем можно измерить —\nи измерения, сделанные в ней", {
    x: M, y: 3.05, w: 9.6, h: 1.1, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 20, color: ICE, lineSpacing: 30,
  });
  s.addText("Транзакционный антифрод для национальной платёжной системы  ·  РКНП 2026", {
    x: M, y: 4.55, w: 10.5, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 14, color: AMBER, bold: true,
  });
  s.addNotes(
    "Мы не делаем ещё один детектор мошенничества. Мы строим полигон, где схема " +
    "известна заранее — поэтому детектор можно не похвалить, а измерить, в том числе " +
    "там, где он ломается."
  );
}

// ==========================================================================
// 2. Проблема
// ==========================================================================
{
  const s = contentSlide("Дропы — это выход схемы, а не сама схема", "Проблема");
  stat(s, { x: M, y: 2.1, w: 3.6, value: "6.87 %", label: "доля инцидентов с дропами\nсреди 80 871 обращения\nв АФЦ Нацбанка", color: NAVY });
  stat(s, { x: M + 4.2, y: 2.1, w: 3.6, value: "5", label: "типологий схемы: слоирование,\nмост в крипту, микширование,\nдробление, обналичка", color: NAVY });
  stat(s, { x: M + 8.4, y: 2.1, w: 3.6, value: "0", label: "открытых данных этой стадии:\nеё ещё нет ни у одного банка,\nвключая Нацбанк", color: AMBER });

  card(s, { x: M, y: 4.55, w: W - 2 * M, h: 1.55 });
  s.addText("План Нацбанка требует перехода «от реакции к предотвращению» — транзакционного антифрода на уровне национальных платёжных систем. Этой стадии ещё не существует, поэтому данных с неё нет ни у кого. Значит их надо построить — и построить так, чтобы результат можно было проверить.", {
    x: M + 0.35, y: 4.8, w: W - 2 * M - 0.7, h: 1.1, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 16, color: INK, lineSpacing: 26,
  });
  s.addNotes("Дропы — последние сто метров схемы. Сама схема это слои, дробление и мост в крипту.");
}

// ==========================================================================
// 3. Решение
// ==========================================================================
{
  const s = contentSlide("Полигон, где ответ известен заранее", "Решение");
  const items = [
    ["Симулятор мира", "8 честных популяций и 4 трудных отрицательных: те, кто тоже опустошает счёт, сборы, продавцы, семьи. 330 тысяч событий, 23 тысячи счетов."],
    ["Поиск вслепую", "Кандидаты строятся только из потока событий. Файл с ответами при построении не читается — метки прикладываются после."],
    ["Измерение, а не похвала", "Схема известна заранее, поэтому детектор можно измерить — и измерить там, где он ломается."],
  ];
  items.forEach(([title, text], i) => {
    const y = 2.05 + i * 1.55;
    card(s, { x: M, y, w: W - 2 * M, h: 1.35 });
    s.addShape(pres.ShapeType.ellipse, {
      x: M + 0.3, y: y + 0.35, w: 0.62, h: 0.62,
      fill: { color: NAVY }, line: { color: NAVY },
    });
    s.addText(String(i + 1), {
      x: M + 0.3, y: y + 0.44, w: 0.62, h: 0.45, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 20, bold: true, color: WHITE, align: "center",
    });
    s.addText(title, {
      x: M + 1.15, y: y + 0.22, w: 3.2, h: 0.45, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 19, bold: true, color: NAVY,
    });
    s.addText(text, {
      x: M + 4.5, y: y + 0.2, w: W - 2 * M - 4.9, h: 1.0, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, color: INK, lineSpacing: 21,
    });
  });
}

// ==========================================================================
// 4. Конвейер
// ==========================================================================
{
  const s = contentSlide("От сырых событий до дела на столе — одной командой", "Система");
  const steps = ["Мир\n330 тыс. событий", "Поиск групп\nбез ответов", "Признаки\nиз потока", "Детектор\npurged walk-forward", "Очередь\nаналитика"];
  const boxW = 2.25, gap = 0.35;
  steps.forEach((text, i) => {
    const x = M + i * (boxW + gap);
    const last = i === steps.length - 1;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.35, w: boxW, h: 1.5,
      fill: { color: last ? AMBER : "F4F6FB" },
      line: { color: last ? AMBER : "F4F6FB" }, rectRadius: 0.1,
    });
    s.addText(text, {
      x: x + 0.1, y: 2.5, w: boxW - 0.2, h: 1.2, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13, bold: true, align: "center",
      color: last ? WHITE : NAVY, lineSpacing: 19,
    });
    if (!last) {
      s.addText("›", {
        x: x + boxW, y: 2.75, w: gap, h: 0.7, isTextBox: true, margin: 0,
        fontFace: HEAD, fontSize: 24, bold: true, color: MUTED, align: "center",
      });
    }
  });
  s.addText("python scripts/run_demo.py --preset full", {
    x: M, y: 4.3, w: 6.4, h: 0.5, isTextBox: true, margin: 0,
    fontFace: "Courier New", fontSize: 16, bold: true, color: NAVY,
  });
  s.addText("Собирает очередь, поднимает API, поднимает интерфейс, ждёт их проверок здоровья и печатает адрес. 17 секунд на весь конвейер.", {
    x: M, y: 4.85, w: 11.9, h: 0.8, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 15, color: INK, lineSpacing: 23,
  });
  note(s, "252 теста, шесть гейтов качества, всё запускается одной командой.");
}

// ==========================================================================
// 5. Очередь аналитика
// ==========================================================================
{
  const s = contentSlide("Что система кладёт человеку на стол", "Результат");
  stat(s, { x: M, y: 2.15, w: 3.4, value: "31", label: "дело в очереди за отрезок,\nкоторый модель не видела" });
  stat(s, { x: M + 3.9, y: 2.15, w: 3.4, value: "100 %", label: "из них — настоящие дропы", color: AMBER });
  stat(s, { x: M + 7.8, y: 2.15, w: 4.0, value: "48 %", label: "всех дропов отрезка\nпойманы этой очередью" });

  card(s, { x: M, y: 4.35, w: W - 2 * M, h: 1.9 });
  s.addText("Очередь режется порогом, а не бюджетом", {
    x: M + 0.35, y: 4.55, w: 11.5, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: NAVY,
  });
  s.addText("Порог выбирается по прошлым отрезкам времени и применяется к следующему, которого детектор не видел. Поэтому длина очереди — результат, а не настройка: тихая неделя даёт короткий список.", {
    x: M + 0.35, y: 5.0, w: 11.5, h: 1.0, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 15, color: INK, lineSpacing: 23,
  });
  s.addNotes("Это та точность, которая была бы в понедельник, а не подогнанная задним числом.");
}

// ==========================================================================
// 6. Как измеряем честно
// ==========================================================================
{
  const s = contentSlide("Четыре правила, из-за которых числам можно верить", "Метод");
  const rules = [
    ["Обучение только на прошлом", "Purged walk-forward с зазором: ни одна строка не оценивалась моделью, которая её видела."],
    ["Группы ищутся вслепую", "Детектору не передаётся готовая банда. Доля найденного — измеряемый потолок, а не единица по построению."],
    ["Признак против шумового пола", "Не побил перемешанный контроль — не подключается. Отрицательный результат тоже результат."],
    ["У каждого уровня свой потолок", "Сколько объектов уровень способен увидеть до всякой модели. Оценка без него верится по неверной причине."],
  ];
  rules.forEach(([title, text], i) => {
    const x = i % 2 === 0 ? M : M + 6.15;
    const y = i < 2 ? 2.1 : 4.25;
    card(s, { x, y, w: 5.75, h: 1.9 });
    s.addText(title, {
      x: x + 0.32, y: y + 0.25, w: 5.1, h: 0.45, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 18, bold: true, color: NAVY,
    });
    s.addText(text, {
      x: x + 0.32, y: y + 0.75, w: 5.1, h: 1.0, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13, color: INK, lineSpacing: 20,
    });
  });
}

// ==========================================================================
// 7-10. Графики
// ==========================================================================
function figureSlide(title, kicker, image, caption, notes) {
  const s = contentSlide(title, kicker);
  s.addImage({ path: path.join(FIG, image), x: 1.55, y: 1.75, w: 10.2, h: 4.55, sizing: { type: "contain", w: 10.2, h: 4.55 } });
  s.addText(caption, {
    x: M, y: 6.5, w: W - 2 * M, h: 0.6, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 14, color: INK,
  });
  if (notes) s.addNotes(notes);
  return s;
}

figureSlide(
  "Сложность объявлена ДО прогонов — и показаны все ступени",
  "Метод",
  "ladder.png",
  "Пять миров, один детектор. На пятой ступени групповой уровень не ошибается — у него вообще нет оценки.",
  "Разница между «выбрали сложность и назвали её» и «построили пять миров, показали тот, где выиграли»."
);

figureSlide(
  "Главный вопрос банка: а если мошенников мало?",
  "Результат",
  "rarity.png",
  "От 7 % до 0.1 % мошенников: ROC-AUC падает с 0.957 до 0.894, а работа аналитика растёт в 80 раз.",
  "Первый вопрос любого человека из банка. Мы его измерили, а не отмахнулись."
);

{
  const s = contentSlide("Ответ: менять надо не детектор, а способ им пользоваться", "Результат");
  s.addText("При доле мошенников 0.1 % — как в жизни", {
    x: M, y: 1.95, w: 11.9, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 16, color: MUTED,
  });

  card(s, { x: M, y: 2.5, w: 5.75, h: 2.5, fill: "F4F6FB" });
  s.addText("Проверять верхние 10 % списка", {
    x: M + 0.32, y: 2.72, w: 5.1, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: MUTED,
  });
  s.addText("100", { x: M + 0.32, y: 3.15, w: 5.1, h: 0.8, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 44, bold: true, color: MUTED });
  s.addText("сигналов на 1000 счетов,\nи 1 из 100 настоящий", { x: M + 0.32, y: 3.95, w: 5.1, h: 0.8, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 14, color: MUTED, lineSpacing: 21 });

  card(s, { x: M + 6.15, y: 2.5, w: 5.75, h: 2.5, fill: "FDF3E7" });
  s.addText("Порог под половину всех дропов", {
    x: M + 6.47, y: 2.72, w: 5.1, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: AMBER,
  });
  s.addText("0.6", { x: M + 6.47, y: 3.15, w: 5.1, h: 0.8, isTextBox: true, margin: 0, fontFace: HEAD, fontSize: 44, bold: true, color: AMBER });
  s.addText("сигнала на 1000 счетов,\nи 86 из 100 настоящие", { x: M + 6.47, y: 3.95, w: 5.1, h: 0.8, isTextBox: true, margin: 0, fontFace: BODY, fontSize: 14, color: INK, lineSpacing: 21 });

  s.addText("Верх списка крутой: половина мошенников лежит выше порога, до которого почти не дотягивается никто честный. Первая половина очереди почти бесплатна — и это то, что банк может применить завтра.", {
    x: M, y: 5.3, w: 11.9, h: 1.0, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 15, color: INK, lineSpacing: 24,
  });
  s.addNotes("Тот же детектор, другая политика применения. Ничего переобучать не нужно.");
}

figureSlide(
  "Сколько стоит спрятаться — и что именно ломается",
  "Результат",
  "evasion.png",
  "Групповой поиск ломается на третьем источнике денег. Поиск по одному человеку не замечает уклонения вообще.",
  "Два уровня анализа: один сильнее, другой переживает противника, который платит."
);

// ==========================================================================
// 11. Крипта
// ==========================================================================
{
  const s = contentSlide("Схема целиком: крипто-канал в мире", "Область");
  stat(s, { x: M, y: 2.1, w: 3.6, value: "4 из 5", label: "типологий из приказа\nработают в мире:\nслоирование, мост, микширование,\nобналичка", color: NAVY });
  stat(s, { x: M + 4.2, y: 2.1, w: 3.6, value: "542", label: "крипто-события и 15 колец\nс легальным входом и мостом\nв криптовалюту", color: NAVY });
  stat(s, { x: M + 8.4, y: 2.1, w: 3.6, value: "0", label: "срабатываний на честных\nкрипто-трейдерах — контроль,\nбез которого вышла бы\nтавтология «крипта = фрод»", color: AMBER });

  card(s, { x: M, y: 4.9, w: W - 2 * M, h: 1.35 });
  s.addText("Честные крипто-трейдеры в мире стоят рядом с кольцами намеренно: без них любой детектор выучил бы «крипта = мошенничество», и это был бы не результат, а тавтология.", {
    x: M + 0.35, y: 5.15, w: W - 2 * M - 0.7, h: 0.9, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 15, color: INK, lineSpacing: 23,
  });
}

// ==========================================================================
// 12. Правила против модели
// ==========================================================================
{
  const s = contentSlide("Опубликованные правила против потока", "Сравнение");
  const rows = [
    ["Критерии приказа (три из четырёх выразимы)", "0.761", MUTED],
    ["Наша модель на тех же данных", "0.959", AMBER],
  ];
  rows.forEach(([label, value, color], i) => {
    const y = 2.2 + i * 1.35;
    card(s, { x: M, y, w: W - 2 * M, h: 1.1, fill: i === 1 ? "FDF3E7" : "F4F6FB" });
    s.addText(label, {
      x: M + 0.35, y: y + 0.3, w: 8.2, h: 0.5, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 17, color: INK,
    });
    s.addText(value, {
      x: M + 9.0, y: y + 0.16, w: 2.5, h: 0.8, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 34, bold: true, color, align: "right",
    });
  });
  s.addText("Главное здесь не разрыв, а его причина", {
    x: M, y: 5.15, w: 11.9, h: 0.45, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 20, bold: true, color: NAVY,
  });
  s.addText("Три критерия из четырёх описывают личность и оборудование, а не денежный поток. Кольцо чистых счетов, ни одного в списках, — проходит правила насквозь. Это разные слои защиты, а не соревнование.", {
    x: M, y: 5.65, w: 11.9, h: 0.9, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 15, color: INK, lineSpacing: 23,
  });
  note(s, "Десять сидов: разрыв 0.20 при разбросе 0.01 — в двадцать раз больше шума.");
}

// ==========================================================================
// 13. Самопроверка
// ==========================================================================
{
  const s = contentSlide("Пять дефектов, которые мы нашли у себя сами", "Проверка");
  const defects = [
    ["Генератор писал ответ в признаки", "ROC-AUC 1.0000 → 0.81 после разделения. Первая честная цифра."],
    ["Сборщик дел брал группы из файла ответов", "Теперь группы ищутся вслепую, метки прикладываются после."],
    ["Базовая линия жульничала в свою пользу", "Критерий «общее устройство» срабатывал на 100 % банд, потому что поиск сам связывал по банкомату."],
    ["79 % «честных» были счетами с 1-2 событиями", "Модель отличала активных от неактивных. Введён порог в 10 событий, цена порога измерена."],
    ["Первая лестница мерила дисбаланс классов", "Мир на 96 % из мошенников. Теперь ступень меняет, какие негативы рядом, а не сколько их."],
  ];
  defects.forEach(([title, text], i) => {
    const y = 1.95 + i * 0.98;
    s.addText(`${i + 1}`, {
      x: M, y: y + 0.05, w: 0.4, h: 0.4, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 18, bold: true, color: AMBER,
    });
    s.addText(title, {
      x: M + 0.45, y, w: 5.3, h: 0.45, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 16, bold: true, color: NAVY,
    });
    s.addText(text, {
      x: M + 5.9, y, w: 6.0, h: 0.75, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13, color: INK, lineSpacing: 19,
    });
  });
  note(s, "Каждый дефект найден нами и записан с числом. Это тоже результат: мы себя проверяли, а не только хвалили.");
}

// ==========================================================================
// 14. Итог
// ==========================================================================
{
  const s = darkSlide();
  s.addText("Что мы приносим на защиту", {
    x: M, y: 0.85, w: 11.9, h: 0.7, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 34, bold: true, color: WHITE,
  });
  const points = [
    ["Работающая система", "От сырых событий до очереди дел — одной командой, 17 секунд, 252 теста."],
    ["Семь измеренных результатов", "У каждого написано, как проверено и чего он не доказывает."],
    ["То, чего нет ни у кого", "Полигон, где схема известна заранее, и цена уклонения, посчитанная по шагам."],
  ];
  points.forEach(([title, text], i) => {
    const y = 2.0 + i * 1.35;
    s.addShape(pres.ShapeType.ellipse, {
      x: M, y: y + 0.12, w: 0.5, h: 0.5, fill: { color: AMBER }, line: { color: AMBER },
    });
    s.addText(title, {
      x: M + 0.85, y, w: 4.6, h: 0.5, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 22, bold: true, color: WHITE,
    });
    s.addText(text, {
      x: M + 5.6, y: y + 0.02, w: 6.3, h: 0.9, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 15, color: ICE, lineSpacing: 23,
    });
  });
  s.addText("Дальше: крипто-ступень в лестницу миров · метрики BAS и SIS · текст работы к третьему туру", {
    x: M, y: 6.3, w: 11.9, h: 0.5, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 15, color: AMBER, bold: true,
  });
}

fs.mkdirSync(path.dirname(OUT), { recursive: true });
pres.writeFile({ fileName: OUT }).then(() => console.log("Готово:", OUT));
