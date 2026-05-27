using System.Collections;
using UnityEngine;

public class AudioLooper : MonoBehaviour
{
    public AudioSource audioSource;
    public float fadeTime = 5f;  // Duration of the fade

    void Start()
    {
        audioSource = GetComponent<AudioSource>();
        audioSource.Play();
        StartCoroutine(PlayLoopWithFade());
    }

    private IEnumerator PlayLoopWithFade()
    {
        float originalVolume = audioSource.volume;
        while (true)
        {
            // Wait until the clip is about to end
            yield return new WaitForSeconds(audioSource.clip.length - fadeTime);
            // Start fade out
            StartCoroutine(FadeAudio(fadeTime, originalVolume, 0));
            // Wait for clip to end and restart
            yield return new WaitForSeconds(fadeTime);
            audioSource.Play();
            // Fade in
            StartCoroutine(FadeAudio(fadeTime, 0, originalVolume));
        }
    }

    private IEnumerator FadeAudio(float duration, float startLevel, float endLevel)
    {
        float currentTime = 0;

        while (currentTime < duration)
        {
            currentTime += Time.deltaTime;
            audioSource.volume = Mathf.Lerp(startLevel, endLevel, currentTime / duration);
            yield return null;
        }
        audioSource.volume = endLevel;
    }
}
