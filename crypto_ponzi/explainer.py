"""
Explainability module for Company-Level Crypto-Ponzi Detection.

Generates human-readable Russian-language explanations of risk assessment results.
"""
from __future__ import annotations

from typing import Any


def generate_explanation(
    metrics: dict[str, float],
    score_result: dict[str, Any],
    company_name: str = "Компания",
) -> str:
    """
    Generate a structured Russian-language explanation paragraph.

    Cites: % от физлиц, % через криптобиржи, dependency_ratio,
    avg_holding_time, concentration_index.
    """
    prob = score_result["probability"]
    level_ru = score_result["risk_level_ru"]
    emoji = score_result["risk_emoji"]

    phys_pct = metrics.get("physical_inflow_ratio", 0.0) * 100
    crypto_pct = metrics.get("crypto_exposure_ratio", 0.0) * 100
    dep_ratio = metrics.get("dependency_ratio", 0.0)
    holding = metrics.get("avg_holding_time", 0.0)
    conc = metrics.get("concentration_index", 0.0) * 100

    # ── Build explanation sections ─────────────────────────────────
    sections: list[str] = []

    # Overall assessment
    sections.append(
        f"{emoji} **Уровень риска: {level_ru}** (вероятность крипто-пирамиды: {prob:.1%})"
    )

    # Structure analysis
    if phys_pct >= 60:
        sections.append(
            f"🔴 **{phys_pct:.0f}%** входящих поступлений — от физических лиц. "
            "Это значительно превышает нормальный уровень для коммерческой деятельности "
            "и характерно для схем привлечения вкладов населения."
        )
    elif phys_pct >= 35:
        sections.append(
            f"🟡 Доля поступлений от физических лиц составляет **{phys_pct:.0f}%**, "
            "что выше типичного для большинства B2B бизнес-моделей."
        )
    else:
        sections.append(
            f"🟢 Доля поступлений от физических лиц — **{phys_pct:.0f}%** — "
            "находится в пределах нормы для стандартного бизнеса."
        )

    # Crypto exposure
    if crypto_pct >= 30:
        sections.append(
            f"🔴 **{crypto_pct:.1f}%** оборота проходит через криптовалютные биржи. "
            "Высокая крипто-экспозиция указывает на активное использование "
            "цифровых активов для перераспределения средств."
        )
    elif crypto_pct >= 10:
        sections.append(
            f"🟡 Крипто-экспозиция составляет **{crypto_pct:.1f}%** оборота — "
            "умеренный уровень, заслуживающий внимания."
        )
    else:
        sections.append(
            f"🟢 Крипто-экспозиция минимальна — **{crypto_pct:.1f}%** оборота."
        )

    # Dependency
    if dep_ratio >= 0.8:
        sections.append(
            f"🔴 Коэффициент зависимости выплат от притока — **{dep_ratio:.2f}**. "
            "Выплаты практически полностью финансируются новыми поступлениями, "
            "что является ключевым признаком пирамидальной схемы."
        )
    elif dep_ratio >= 0.5:
        sections.append(
            f"🟡 Коэффициент зависимости выплат — **{dep_ratio:.2f}** — "
            "указывает на существенную зависимость от притока."
        )
    else:
        sections.append(
            f"🟢 Коэффициент зависимости выплат — **{dep_ratio:.2f}** — "
            "в допустимых пределах."
        )

    # Holding time
    if holding < 30:
        sections.append(
            f"🔴 Среднее время удержания средств — **{holding:.0f} дней**. "
            "Быстрый оборот средств характерен для схем «приток → вывод → выплаты»."
        )
    elif holding < 60:
        sections.append(
            f"🟡 Среднее удержание средств — **{holding:.0f} дней** — "
            "ниже среднерыночного."
        )
    else:
        sections.append(
            f"🟢 Среднее удержание — **{holding:.0f} дней** — "
            "в рамках нормальной бизнес-деятельности."
        )

    # Concentration
    if conc >= 40:
        sections.append(
            f"🔴 Топ-5 контрагентов генерируют **{conc:.0f}%** оборота. "
            "Высокая концентрация выплат через узкий круг контрагентов."
        )
    elif conc >= 25:
        sections.append(
            f"🟡 Концентрация на топ-5 контрагентов — **{conc:.0f}%** — "
            "умеренно повышена."
        )
    else:
        sections.append(
            f"🟢 Концентрация на топ-5 контрагентов — **{conc:.0f}%** — "
            "разумная диверсификация."
        )

    # Conclusion
    if prob >= 0.75:
        sections.append(
            "---\n"
            f"**Заключение**: {company_name} демонстрирует структурные признаки, "
            "характерные для крипто-ориентированных финансовых перераспределительных схем. "
            "Рекомендуется приоритетная проверка."
        )
    elif prob >= 0.55:
        sections.append(
            "---\n"
            f"**Заключение**: {company_name} демонстрирует выраженные аномалии, "
            "характерные для высокорисковых инвестиционных структур. "
            "Рекомендуется углубленный анализ."
        )
    elif prob >= 0.30:
        sections.append(
            "---\n"
            f"**Заключение**: {company_name} имеет умеренные индикаторы риска. "
            "Рекомендуется включить в план мониторинга."
        )
    else:
        sections.append(
            "---\n"
            f"**Заключение**: {company_name} демонстрирует стандартные показатели "
            "деловой активности. Текущий уровень риска не требует "
            "приоритетной проверки."
        )

    return "\n\n".join(sections)
