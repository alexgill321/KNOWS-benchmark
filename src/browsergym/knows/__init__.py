try:
    from browsergym.core.registration import register_task
    _HAS_BROWSERGYM = True
except ImportError:
    _HAS_BROWSERGYM = False

from .task import (
    DocsFormalLetterTask,
    DocsEducationLessonPlanTask,
    DocsInfluentialPapersTask,
    DocsPersonalRecipeOcrTask,
    DocsReferenceListTask,
    KnowsBenchTask,
    KnowsWorkspaceTask,
    SheetsApartmentFinderTask,
    SheetsMovieRecommendationTask,
    SheetsPaperSortingTask,
    SheetsPersonalRecipeTask,
    SheetsPersonalTravelPlannerTask,
    SheetsRunningAnalysisTask,
    SheetsSkiTourPlanTask,
    SheetsStockTrackerTask,
    SheetsWeddingPlannerTask,
    SlidesBasicEducationalSlideDeckTask,
    SlidesBuyCarPresTask,
    SlidesEventAnnouncementPosterTask,
    SlidesIllustratedBookReportTask,
    SlidesPersonalLookbookPaintColorsTask,
    SlidesProductComparisonTask,
    SlidesRemoveImagesAddPlaceholdersTask,
    SlidesWikipediaPhotosTask,
)

if _HAS_BROWSERGYM:
    # Backwards-compatible alias for the original single-instance task id used by
    # the legacy ``knows_1`` benchmark (mapped to instance_1).
    register_task(
        "knows.docs_1_formal_letter",
        DocsFormalLetterTask,
        task_kwargs={
            "instance_id": 1,
            "task_name": "docs_1_formal_letter/instance_1",
        },
    )

    def _register_task_family(task_cls: type) -> list[str]:
        """Register one gym task per bundled instance for *task_cls*.

        Returns the list of registered gym task ids, e.g.
        ``["knows.docs_1_formal_letter.1", ...]``.
        """
        prefix = task_cls.TASK_ID_PREFIX
        folder = task_cls.TASK_FAMILY_FOLDER
        ids: list[str] = []
        for instance_id in task_cls.AVAILABLE_INSTANCES:
            gym_id = f"{prefix}.{instance_id}"
            register_task(
                gym_id,
                task_cls,
                task_kwargs={
                    "instance_id": instance_id,
                    "task_name": f"{folder}/instance_{instance_id}",
                },
            )
            ids.append(gym_id)
        return ids

    # Register one gym task per instance, e.g. "knows.docs_1_formal_letter.1".
    DOCS_1_FORMAL_LETTER_INSTANCES = DocsFormalLetterTask.AVAILABLE_INSTANCES
    KNOWS_DOCS_1_TASK_IDS = _register_task_family(DocsFormalLetterTask)

    # All task families — each registers ``knows.<family>.{1..N}`` gym ids.
    KNOWS_SHEETS_2_TASK_IDS = _register_task_family(SheetsPersonalRecipeTask)
    KNOWS_DOCS_5_TASK_IDS = _register_task_family(DocsInfluentialPapersTask)
    KNOWS_SHEETS_6_TASK_IDS = _register_task_family(SheetsStockTrackerTask)
    KNOWS_SHEETS_7_TASK_IDS = _register_task_family(SheetsRunningAnalysisTask)
    KNOWS_SHEETS_10_TASK_IDS = _register_task_family(SheetsPaperSortingTask)
    KNOWS_DOCS_11_TASK_IDS = _register_task_family(DocsPersonalRecipeOcrTask)
    KNOWS_SLIDES_17_TASK_IDS = _register_task_family(SlidesRemoveImagesAddPlaceholdersTask)
    KNOWS_SLIDES_20_TASK_IDS = _register_task_family(SlidesIllustratedBookReportTask)
    KNOWS_SHEETS_25_TASK_IDS = _register_task_family(SheetsSkiTourPlanTask)
    KNOWS_SLIDES_26_TASK_IDS = _register_task_family(SlidesBasicEducationalSlideDeckTask)
    KNOWS_SLIDES_25_TASK_IDS = KNOWS_SLIDES_26_TASK_IDS
    KNOWS_SHEETS_28_TASK_IDS = _register_task_family(SheetsPersonalTravelPlannerTask)
    KNOWS_SLIDES_29_TASK_IDS = _register_task_family(SlidesBuyCarPresTask)
    KNOWS_SLIDES_30_TASK_IDS = _register_task_family(SlidesWikipediaPhotosTask)
    KNOWS_DOCS_31_TASK_IDS = _register_task_family(DocsEducationLessonPlanTask)
    KNOWS_DOCS_37_TASK_IDS = _register_task_family(DocsReferenceListTask)
    KNOWS_SHEETS_38_TASK_IDS = _register_task_family(SheetsApartmentFinderTask)
    KNOWS_SLIDES_39_TASK_IDS = _register_task_family(SlidesPersonalLookbookPaintColorsTask)
    KNOWS_SLIDES_42_TASK_IDS = _register_task_family(SlidesProductComparisonTask)
    KNOWS_SHEETS_45_TASK_IDS = _register_task_family(SheetsWeddingPlannerTask)
    KNOWS_SLIDES_51_TASK_IDS = _register_task_family(SlidesEventAnnouncementPosterTask)
    KNOWS_SHEETS_55_TASK_IDS = _register_task_family(SheetsMovieRecommendationTask)


__all__ = [
    "DocsFormalLetterTask",
    "DocsEducationLessonPlanTask",
    "DocsInfluentialPapersTask",
    "DocsPersonalRecipeOcrTask",
    "DocsReferenceListTask",
    "DOCS_1_FORMAL_LETTER_INSTANCES",
    "KnowsBenchTask",
    "KnowsWorkspaceTask",
    "KNOWS_DOCS_1_TASK_IDS",
    "KNOWS_DOCS_5_TASK_IDS",
    "KNOWS_DOCS_11_TASK_IDS",
    "KNOWS_DOCS_31_TASK_IDS",
    "KNOWS_DOCS_37_TASK_IDS",
    "KNOWS_SHEETS_2_TASK_IDS",
    "KNOWS_SHEETS_6_TASK_IDS",
    "KNOWS_SHEETS_7_TASK_IDS",
    "KNOWS_SHEETS_10_TASK_IDS",
    "KNOWS_SHEETS_25_TASK_IDS",
    "KNOWS_SHEETS_28_TASK_IDS",
    "KNOWS_SHEETS_38_TASK_IDS",
    "KNOWS_SHEETS_45_TASK_IDS",
    "KNOWS_SHEETS_55_TASK_IDS",
    "KNOWS_SLIDES_17_TASK_IDS",
    "KNOWS_SLIDES_20_TASK_IDS",
    "KNOWS_SLIDES_25_TASK_IDS",
    "KNOWS_SLIDES_26_TASK_IDS",
    "KNOWS_SLIDES_29_TASK_IDS",
    "KNOWS_SLIDES_30_TASK_IDS",
    "KNOWS_SLIDES_39_TASK_IDS",
    "KNOWS_SLIDES_42_TASK_IDS",
    "KNOWS_SLIDES_51_TASK_IDS",
    "SheetsApartmentFinderTask",
    "SheetsMovieRecommendationTask",
    "SheetsPaperSortingTask",
    "SheetsPersonalRecipeTask",
    "SheetsPersonalTravelPlannerTask",
    "SheetsRunningAnalysisTask",
    "SheetsSkiTourPlanTask",
    "SheetsStockTrackerTask",
    "SheetsWeddingPlannerTask",
    "SlidesBasicEducationalSlideDeckTask",
    "SlidesBuyCarPresTask",
    "SlidesEventAnnouncementPosterTask",
    "SlidesIllustratedBookReportTask",
    "SlidesPersonalLookbookPaintColorsTask",
    "SlidesProductComparisonTask",
    "SlidesRemoveImagesAddPlaceholdersTask",
    "SlidesWikipediaPhotosTask",
]
