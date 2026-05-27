using UnityEngine;
using UnityEngine.UI;

public class ProgressBarManager : MonoBehaviour
{
    public Image progressBarFill; // Assign this in the Inspector
    public Image progressBarBackground; // Assign this in the Inspector
    private float holdTime = 5.0f; // Total time to hold the button
    private float currentHoldTime = 0f;
    private bool isButtonHeld = false;

    void Awake()
    {
        progressBarFill.gameObject.SetActive(false);
        progressBarBackground.gameObject.SetActive(false);
        AlignBackgroundWithFill();
    }
    
    void Update()
    {
        if (isButtonHeld)
        {
            currentHoldTime += Time.deltaTime;
            UpdateProgressBar(currentHoldTime / holdTime);
        }
    }

    public void StartButtonHold()
    {
        isButtonHeld = true;
        currentHoldTime = 0f;
        progressBarFill.gameObject.SetActive(true);
        progressBarBackground.gameObject.SetActive(true);
    }

    public void EndButtonHold()
    {
        isButtonHeld = false;
        progressBarFill.gameObject.SetActive(false);
        progressBarBackground.gameObject.SetActive(false);
    }

    void UpdateProgressBar(float progress)
    {
        progressBarFill.fillAmount = progress;
    }

    private void AlignBackgroundWithFill()
    {
        if (progressBarFill != null && progressBarBackground != null)
        {
            RectTransform fillTransform = progressBarFill.GetComponent<RectTransform>();
            RectTransform backgroundTransform = progressBarBackground.GetComponent<RectTransform>();

            backgroundTransform.position = fillTransform.position;
            backgroundTransform.sizeDelta = fillTransform.sizeDelta;
            backgroundTransform.anchorMin = fillTransform.anchorMin;
            backgroundTransform.anchorMax = fillTransform.anchorMax;
            backgroundTransform.pivot = fillTransform.pivot;

            // Ensure the background is always rendered behind the fill
            progressBarBackground.transform.SetSiblingIndex(0);
            progressBarFill.transform.SetSiblingIndex(1);
        }
    }
}
